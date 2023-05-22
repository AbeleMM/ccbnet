import warnings
from collections import Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from random import Random
from time import perf_counter_ns
from typing import Collection, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from avg_outs import AvgOuts
from combine import CombineMethod, combine_bns
from joblib import Memory
from matplotlib.axes import Axes
from model import Model
from party import combine
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from prod_outs import ProdOuts
from single_net import SingleNet

_memory = Memory(Path(__file__).parents[1] / "cache", verbose=0)
_BENCHMARK_INDEX: str = "Overlap"
BENCHMARK_PIVOT_COL: str = "Name"


def print_bn(bayes_net: BayesianNetwork, struct_only=False) -> None:
    if not struct_only:
        for cpd in bayes_net.get_cpds() or []:
            print(cpd)
            print()
    bayes_net.to_daft().render()


def get_in_out_nodes(bayes_net: BayesianNetwork) -> tuple[list[str], list[str]]:
    # Island nodes are not included in neither the in, nor the out list.
    in_nodes: list[str] = []
    out_nodes: list[str] = []
    edges_to, edges_from = [set(edge_list) for edge_list in zip(*bayes_net.edges())]

    for node in cast(Collection[str], bayes_net.nodes()):
        if node not in edges_to and node in edges_from:
            out_nodes.append(node)
        elif node not in edges_from and node in edges_to:
            in_nodes.append(node)

    return (in_nodes, out_nodes)


def split_vars(
        bayes_net: BayesianNetwork,
        nr_splits: int,
        overlap_proportion: float,
        connected=True,
        seed: int | None = None) -> list[list[str]]:
    rand = Random(seed)  # nosec
    nr_overlaps = round(overlap_proportion * len(bayes_net.nodes()))

    if connected:
        dfs_tree = nx.dfs_tree(bayes_net.to_undirected())
        communities: list[set[str]] = [
            set(community)
            for community in cast(
                list[frozenset[str]],
                nx.algorithms.community.greedy_modularity_communities(
                    dfs_tree,
                    cutoff=nr_splits,
                    best_n=nr_splits
                )
            )
        ]
    else:
        shuffled_nodes: list[str] = list(bayes_net.nodes())
        rand.shuffle(shuffled_nodes)
        communities: list[set[str]] = [set(x) for x in np.array_split(shuffled_nodes, nr_splits)]

    shuffled_edges = cast(list[tuple[str, str]], list(bayes_net.edges()))
    rand.shuffle(shuffled_edges)
    return _overlap_communities(communities, nr_overlaps, shuffled_edges)


@_memory.cache
def train_model(samples: pd.DataFrame, max_nr_parents: int, states: dict[str, list[str]]) -> \
        BayesianNetwork:
    est = HillClimbSearch(data=samples, use_cache=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        estimated_dag = est.estimate(
            scoring_method="k2score",
            max_indegree=max_nr_parents,
            max_iter=10**4,
            show_progress=False
        )

    bayes_net = BayesianNetwork()
    bayes_net.add_nodes_from(samples.columns)
    bayes_net.add_edges_from(estimated_dag.edges())

    bayes_net.fit(
        samples,
        estimator=BayesianEstimator,
        state_names=states,
        complete_samples_only=True
    )

    return bayes_net


def get_inf_res(
        source: Model,
        samples: list[dict[str, str]],
        q_vars: list[str]) -> tuple[list[DiscreteFactor], float]:
    facts: list[DiscreteFactor] = []
    total_time = 0.0

    for i, row in enumerate(samples):
        start = perf_counter_ns()
        query = source.query([q_vars[i]], row)

        total_time += perf_counter_ns() - start
        facts.append(query)

    return facts, total_time


def calc_brier(
        ref_facts: list[DiscreteFactor],
        pred_facts: list[DiscreteFactor]) -> float:
    acc = 0.0

    for i, ref_fact in enumerate(ref_facts):

        for node in ref_fact.variables:
            for state in ref_fact.state_names[node]:
                node_state = {node: state}
                ref_val = cast(float, ref_fact.get_value(**node_state))
                try:
                    pred_val = cast(float, pred_facts[i].get_value(**node_state))
                except ValueError:
                    pred_val = 0.0
                acc += (ref_val - pred_val) ** 2

    return acc / len(ref_facts)


def benchmark_single(
        ref_model: BayesianNetwork,
        trained_models: list[BayesianNetwork],
        test_samples: list[dict[str, str]],
        query_vars: list[str],
        learnt_model: BayesianNetwork | None = None) -> pd.DataFrame:
    res_single: list[dict[str, float | str]] = []
    name_to_bn: dict[str, Model] = {}
    ref_facts, total_ref_time = get_inf_res(SingleNet.from_bn(ref_model), test_samples, query_vars)

    if learnt_model:
        name_to_bn["Learnt"] = SingleNet.from_bn(learnt_model)

    # # TODO switch to combine_bns_weighted (by amount of samples)
    # for method in CombineMethod:
    #     name_to_bn[method.value] = combine_bns(trained_models, method)

    name_to_bn["Combine"] = combine_bns(trained_models, CombineMethod.MULTI)
    name_to_bn["Union"] = combine_bns(trained_models, CombineMethod.UNION)

    name_to_bn["Decentralized"] = combine(trained_models)

    name_to_bn["AvgOuts"] = AvgOuts(trained_models)
    name_to_bn["ProdOuts"] = ProdOuts(trained_models)

    for name, model in name_to_bn.items():
        row: dict[str, float | str] = {BENCHMARK_PIVOT_COL: name}
        pred_facts, total_pred_time = get_inf_res(model, test_samples, query_vars)

        row["Brier"] = round(calc_brier(ref_facts, pred_facts), 3)

        row["RelTotTime"] = round(total_pred_time / total_ref_time, 2)

        # row["StructureF1"] = round(sf1_score(ref_model, model), 3)

        # row["SHD"] = shd_score(ref_model, model)

        # row["EdgeCount"] = len(model.edges())

        res_single.append(row)

    return pd.DataFrame.from_records(res_single)


def benchmark_multi(
        ref_model: BayesianNetwork,
        nr_clients: int,
        overlap_ratios: list[float],
        samples_factor: int = 50000,
        test_counts: int = 2000,
        include_learnt=False,
        in_out_inf_vars=True,
        rand_inf_vars=True,
        r_seed: int | None = None) -> dict[str, pd.DataFrame]:
    samples_per_client = samples_factor * len(ref_model.nodes()) // nr_clients
    scen_to_df: dict[str, pd.DataFrame] = {}
    scen_to_q_vars: dict[str, list[str]] = {}
    scen_to_test_insts: dict[str, list[dict[str, str]]] = {}
    # if in_out_inf_vars:
    #     scen_to_df["inout"] = pd.DataFrame()
    # if rand_inf_vars:
    #     scen_to_df["rand"] = pd.DataFrame()
    scen_to_df["mix"] = pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        all_samples = BayesianModelSampling(ref_model).forward_sample(
            size=samples_per_client * nr_clients + test_counts,
            seed=r_seed,
            show_progress=False
        )

        clients_train_samples = [
            all_samples[i * samples_per_client:(i + 1) * samples_per_client]
            for i in range(nr_clients)
        ]

        test_samples = cast(list[dict[str, str]],
                            all_samples[-test_counts:].to_dict(orient="records"))

        scen_to_test_insts["mix"] = []
        scen_to_q_vars["mix"] = []
        rand = Random(r_seed)

        for test_sample in test_samples:
            evid = rand.sample(ref_model.nodes(), round(0.2 * len(ref_model)))
            scen_to_test_insts["mix"].append({k: test_sample[k] for k in evid})
            scen_to_q_vars["mix"].append(
                rand.choice([n for n in ref_model.nodes() if n not in evid]))

        # if in_out_inf_vars:
        #     evidence_vars, query_vars = get_in_out_nodes(ref_model)
        #     scen_to_q_vars["inout"] = query_vars
        #     scen_to_test_insts["inout"] = cast(
        #         list[dict[str, str]], test_samples[evidence_vars].to_dict(orient="records"))

        # if rand_inf_vars:
        #     rand = Random(r_seed)  # nosec
        #     shuffled_nodes: list[str] = list(ref_model.nodes())
        #     rand.shuffle(shuffled_nodes)
        #     mid_ind = len(shuffled_nodes) // 2
        #     evidence_vars, query_vars = shuffled_nodes[:mid_ind], shuffled_nodes[mid_ind:]
        #     scen_to_q_vars["rand"] = query_vars
        #     scen_to_test_insts["rand"] = cast(
        #         list[dict[str, str]], test_samples[evidence_vars].to_dict(orient="records"))

        (_, max_in_deg), *_ = Counter(e_inc for (_, e_inc, *_) in ref_model.edges()).most_common(1)

        if include_learnt:
            samples = pd.concat(clients_train_samples, ignore_index=True, copy=False)
            learnt_model = train_model(samples, max_in_deg, ref_model.states)
        else:
            learnt_model = None

        for overlap in overlap_ratios:
            print(overlap)
            clients_train_vars = split_vars(ref_model, nr_clients, overlap, True, seed=r_seed)

            trained_models = [
                train_model(clients_train_samples[i][train_vars], max_in_deg, ref_model.states)
                for i, train_vars in enumerate(clients_train_vars)
            ]

            for scen, d_f in scen_to_df.items():
                res_single = benchmark_single(
                    ref_model, trained_models, scen_to_test_insts[scen], scen_to_q_vars[scen],
                    learnt_model if not overlap else None)
                res_single[_BENCHMARK_INDEX] = round(overlap, 1)
                scen_to_df[scen] = pd.concat([d_f, res_single], ignore_index=True, copy=False)

    for d_f in scen_to_df.values():
        d_f.set_index(_BENCHMARK_INDEX, inplace=True)

    return scen_to_df


def _overlap_communities(
        community_sets:  list[set[str]],
        nr_overlaps: int,
        shuffled_edges: list[tuple[str, str]]) -> list[list[str]]:
    node_to_community: dict[str, int] = {}
    overlap_nodes: set[str] = set()
    overlapped_communities: set[int] = set()
    remaining_edges: list[tuple[str, str]] = []

    for i, community in enumerate(community_sets):
        node_to_community.update({node: i for node in community})

    for node_out, node_inc in shuffled_edges:
        if len(overlap_nodes) >= nr_overlaps:
            break

        community_node_out = node_to_community[node_out]
        community_node_inc = node_to_community[node_inc]

        if community_node_out == community_node_inc:
            continue

        edge_nodes_communities = set([community_node_out, community_node_inc])

        if (
            len(overlapped_communities) < len(community_sets) and
            edge_nodes_communities <= overlapped_communities
        ):
            remaining_edges.append((node_out, node_inc))
        else:
            overlap_nodes.add(node_out)
            overlap_nodes.add(node_inc)
            overlapped_communities.update(edge_nodes_communities)
            community_sets[community_node_out].add(node_inc)
            community_sets[community_node_inc].add(node_out)

    for node_out, node_inc in remaining_edges:
        if len(overlap_nodes) >= nr_overlaps:
            break

        overlap_nodes.add(node_out)
        overlap_nodes.add(node_inc)
        community_sets[node_to_community[node_out]].add(node_inc)
        community_sets[node_to_community[node_inc]].add(node_out)

    return [sorted(community) for community in community_sets]


class ExpWriter():
    def __init__(self) -> None:
        time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.res_dir = Path(__file__).parents[1] / "out" / time_str
        self.res_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(self, axes: Axes, name: str) -> None:
        fig = axes.get_figure()
        fig.savefig(str((self.res_dir / name).with_suffix(".png")), bbox_inches="tight")
        plt.close(fig)
