import warnings
from collections import Counter
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from random import Random
from time import perf_counter_ns
from typing import Collection, cast

import inference
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from combine import CombineMethod, combine_bns
from decentralized.client import Client, combine
from joblib import Memory
from matplotlib.axes import Axes
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.metrics import f1_score, mean_squared_error

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
        split_size = ceil(len(shuffled_nodes) / nr_splits)
        communities = [
            set(shuffled_nodes[i:i + split_size])
            for i in range(0, len(shuffled_nodes), split_size)
        ]

    shuffled_edges = cast(list[tuple[str, str]], list(bayes_net.edges()))
    rand.shuffle(shuffled_edges)
    return _overlap_communities(communities, nr_overlaps, shuffled_edges)


@_memory.cache
def train_model(samples: pd.DataFrame, max_nr_parents: int) -> BayesianNetwork:
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
        complete_samples_only=True
    )

    return bayes_net


def get_inf_res(
        source: BayesianNetwork | Client,
        samples: list[dict[str, str]],
        q_vars: list[str]) -> tuple[list[dict[str, DiscreteFactor]], float]:
    facts: list[dict[str, DiscreteFactor]] = []
    total_time = 0.0

    for row in samples:
        if isinstance(source, BayesianNetwork):
            start = perf_counter_ns()
            query = inference.disjoint_elimination_ask(source, q_vars, row)
        else:
            start = perf_counter_ns()
            query = source.disjoint_elimination_ask(q_vars, row)

        total_time += perf_counter_ns() - start
        facts.append(query)

    return facts, total_time


def calc_rmse(
        ref_fact_dicts: list[dict[str, DiscreteFactor]],
        pred_fact_dicts: list[dict[str, DiscreteFactor]]) -> float:
    ref_values: list[list[float]] = []
    pred_values: list[list[float]] = []

    for i, ref_fact_dict in enumerate(ref_fact_dicts):
        ref_row: list[float] = []
        pred_row: list[float] = []

        for node, ref_fact in ref_fact_dict.items():
            for state in ref_fact.state_names[node]:
                node_state = {node: state}
                ref_row.append(cast(float, ref_fact.get_value(**node_state)))
                try:
                    pred_val = cast(float, pred_fact_dicts[i][node].get_value(**node_state))
                except ValueError:
                    pred_val = 0.0
                pred_row.append(pred_val)

        ref_values.append(ref_row)
        pred_values.append(pred_row)

    return cast(float, mean_squared_error(ref_values, pred_values, squared=False))


def benchmark_single(
        ref_model: BayesianNetwork,
        trained_models: list[BayesianNetwork],
        test_samples: list[dict[str, str]],
        query_vars: list[str],
        learnt_model: BayesianNetwork | None = None,
        decentralized=False) -> pd.DataFrame:
    res_single: list[dict[str, float | str]] = []
    name_to_bn: dict[str, BayesianNetwork | Client] = {}
    ref_facts, total_ref_time = get_inf_res(ref_model, test_samples, query_vars)

    if learnt_model:
        name_to_bn["Learnt"] = learnt_model

    # for method in CombineMethod:
    #     # TODO switch to combine_bns_weighted (by amount of samples)
    #     name_to_bn[method.value] = VariableElimination(combine_bns(trained_models, method))

    name_to_bn["Combine"] = combine_bns(trained_models, CombineMethod.MULTI)
    name_to_bn["Union"] = combine_bns(trained_models, CombineMethod.UNION)

    if decentralized:
        name_to_bn["Decentralized"] = combine(trained_models)

    for name, model in name_to_bn.items():
        row: dict[str, float | str] = {BENCHMARK_PIVOT_COL: name}
        pred_facts, total_pred_time = get_inf_res(model, test_samples, query_vars)

        row["RMSE"] = round(calc_rmse(ref_facts, pred_facts), 3)

        row["RelTotTime"] = round(total_pred_time / total_ref_time, 2)

        # if not decentralized:
        #     row["StructureF1"] = round(sf1_score(ref_model, model), 3)

        #     row["SHD"] = round(shd_score(ref_model, model), 3)

        #     row["EdgeCount"] = len(model.edges())

        res_single.append(row)

    return pd.DataFrame.from_records(res_single)


def benchmark_multi(
        ref_model: BayesianNetwork,
        nr_clients: int,
        test_counts: int = 2000,
        samples_factor: int = 50000,
        include_learnt=False,
        in_out_inf_vars=True,
        rand_inf_vars=True,
        decentralized=False,
        r_seed: int | None = None) -> dict[str, pd.DataFrame]:
    sampling = BayesianModelSampling(ref_model)
    samples_per_client = samples_factor * len(ref_model.nodes()) // nr_clients
    scen_to_df: dict[str, pd.DataFrame] = {}
    scen_to_q_vars: dict[str, list[str]] = {}
    scen_to_test_insts: dict[str, list[dict[str, str]]] = {}
    if in_out_inf_vars:
        scen_to_df["inout"] = pd.DataFrame()
    if rand_inf_vars:
        scen_to_df["rand"] = pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clients_train_samples = [
            sampling.forward_sample(
                size=samples_per_client,
                seed=r_seed,
                show_progress=False
            )
            for _ in range(nr_clients)
        ]
        (_, max_in_deg), *_ = Counter(e_inc for (_, e_inc, *_) in ref_model.edges()).most_common(1)

        test_samples = sampling.forward_sample(size=test_counts, seed=r_seed, show_progress=False)

        if in_out_inf_vars:
            evidence_vars, query_vars = get_in_out_nodes(ref_model)
            scen_to_q_vars["inout"] = query_vars
            scen_to_test_insts["inout"] = cast(
                list[dict[str, str]], test_samples[evidence_vars].to_dict(orient="records"))

        if rand_inf_vars:
            rand = Random(r_seed)  # nosec
            shuffled_nodes: list[str] = list(ref_model.nodes())
            rand.shuffle(shuffled_nodes)
            mid_ind = len(shuffled_nodes) // 2
            evidence_vars, query_vars = shuffled_nodes[:mid_ind], shuffled_nodes[mid_ind:]
            scen_to_q_vars["rand"] = query_vars
            scen_to_test_insts["rand"] = cast(
                list[dict[str, str]], test_samples[evidence_vars].to_dict(orient="records"))

        if include_learnt:
            samples = pd.concat(clients_train_samples, ignore_index=True, copy=False)
            learnt_model = train_model(samples, max_in_deg)
        else:
            learnt_model = None

        for overlap in np.arange(0.0, 0.6, 0.1):
            clients_train_vars = split_vars(ref_model, nr_clients, overlap, seed=r_seed)

            trained_models = [
                train_model(clients_train_samples[i][train_vars], max_in_deg)
                for i, train_vars in enumerate(clients_train_vars)
            ]

            for scen, d_f in scen_to_df.items():
                res_single = benchmark_single(
                    ref_model, trained_models, scen_to_test_insts[scen], scen_to_q_vars[scen],
                    learnt_model if not overlap else None, decentralized
                )
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


def _bn_to_adj_mat(model: BayesianNetwork, preserve_dir: bool) -> np.ndarray:
    model_transformed = model if preserve_dir else model.to_undirected()
    return nx.convert_matrix.to_numpy_array(model_transformed, nodelist=model.nodes(), weight="")


def sf1_score(true_model: BayesianNetwork, estimated_model: BayesianNetwork) -> float:
    true_adj = _bn_to_adj_mat(true_model, False)
    estimated_adj = _bn_to_adj_mat(estimated_model, False)

    return cast(float, f1_score(np.ravel(true_adj), np.ravel(estimated_adj)))


# https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/master/cdt/metrics.py
def shd_score(
        true_model: BayesianNetwork,
        estimated_model: BayesianNetwork,
        double_for_anticausal=True) -> float:
    true_adj = _bn_to_adj_mat(true_model, True)
    estimated_adj = _bn_to_adj_mat(estimated_model, True)

    diff = np.abs(true_adj - estimated_adj)

    if double_for_anticausal:
        return np.sum(diff)

    diff = diff + diff.transpose()
    diff[diff > 1] = 1

    return float(np.sum(diff) / 2)


class ExpWriter():
    def __init__(self) -> None:
        time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        self.res_dir = Path(__file__).parents[1] / "out" / time_str
        self.res_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(self, axes: Axes, name: str) -> None:
        fig = axes.get_figure()
        fig.savefig(str((self.res_dir / name).with_suffix(".png")), bbox_inches="tight")
        plt.close(fig)
