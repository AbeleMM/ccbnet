import logging
import os
import warnings
from collections import Counter
from collections.abc import Collection, Generator
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from random import Random
from time import perf_counter_ns
from typing import cast

import networkx as nx
import numpy as np
import pandas as pd
from avg_outs import AvgOuts, MeanType
from combine import CombineMethod, CombineOp, combine_bns
from disc_fact import DiscFactCfg
from joblib import Memory, Parallel, delayed
from model import Model
from party import combine
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from single_net import SingleNet
from tqdm import tqdm
from var_elim_heurs import MinWeightVEH

_memory = Memory(Path(__file__).parents[1] / "cache", verbose=0)
_BENCHMARK_INDEX: str = "Overlap"
BENCHMARK_PIVOT_COL: str = "Name"
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)
try:
    N_PARALLEL = len(os.sched_getaffinity(0))  # type: ignore
except AttributeError:
    N_PARALLEL = cast(int, os.cpu_count())
N_PARALLEL -= 1
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"


@dataclass
class TestOut:
    res_facts: list[DiscreteFactor] = field(default_factory=list)
    tot_time: float = field(default=0.0)
    avg_comm_vals: float = field(default=0.0)


def _yield_approaches(trained_models: list[BayesianNetwork], weights: list[float]) -> \
        Generator[tuple[str, Model], None, None]:
    dfc = DiscFactCfg(False, np.float_)
    eq_weights = [1.0] * len(trained_models)
    veh = MinWeightVEH()

    yield "Combine", combine_bns(
        trained_models, eq_weights, CombineMethod.MULTI, True, CombineOp.SUPERPOS, dfc, veh)
    yield "Union", combine_bns(
        trained_models, weights, CombineMethod.UNION, True, CombineOp.GEO_MEAN, dfc, veh)
    # yield "Union - EQ", combine_bns(
    #     trained_models, eq_weights, CombineMethod.UNION, True, CombineOp.GEO_MEAN, dfc, veh)
    yield "AvgOuts", AvgOuts(trained_models, MeanType.GEO, dfc, veh)
    yield "Decentralized", combine(trained_models, weights, True, dfc, veh)
    # yield "Decentralized - EQ", combine(trained_models, eq_weights, True, dfc, veh)
    yield "Decentralized - Compact", combine(trained_models, weights, False, dfc, veh)


def benchmark_multi(
        ref_bn: BayesianNetwork,
        nr_clients: int,
        overlap_ratios: list[float],
        samples_factor: int = 500,
        test_counts: int = 2000,
        connected: bool = True,
        eq_weights: bool = True,
        r_seed: int | None = None) -> pd.DataFrame:
    LOGGER.info("%s %s", ref_bn.name, nr_clients)
    base_samples_per_client = samples_factor * len(ref_bn.nodes()) // nr_clients
    if eq_weights:
        weights = [1.0] * nr_clients
    else:
        rand = Random(r_seed)
        weights = [rand.uniform(0.1, 1.0) for _ in range(nr_clients)]
    nr_samples_per_client = [
        round(base_samples_per_client * weight) for weight in weights]
    nr_evid_vars = round(0.6 * (len(ref_bn) - 1))
    ref_model = SingleNet.from_bn(ref_bn, False, DiscFactCfg(False, np.float_), MinWeightVEH())
    d_f = pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        all_samples = sample(ref_bn, sum(nr_samples_per_client) + test_counts, r_seed)
        clients_train_samples: list[pd.DataFrame] = []
        running_sum = 0
        for nr_client_samples in nr_samples_per_client:
            clients_train_samples.append(all_samples[running_sum:running_sum + nr_client_samples])
            running_sum += nr_client_samples
        test_samples = cast(
            list[dict[str, str]], all_samples[-test_counts:].to_dict(orient="records"))
        LOGGER.info("Sampled")
        (_, max_in_deg), *_ = Counter(e_inc for (_, e_inc, *_) in ref_bn.edges()).most_common(1)

        for overlap in overlap_ratios:
            LOGGER.info("Overlap %s", overlap)
            clients_train_vars, ov_vars = _split_vars(
                ref_bn, nr_clients, overlap, connected, seed=r_seed)
            LOGGER.info("Training models")
            trained_models = cast(list[BayesianNetwork], Parallel(N_PARALLEL)(
                delayed(_train_model)
                (clients_train_samples[i][train_vars], max_in_deg, ref_bn.states)
                for i, train_vars in enumerate(clients_train_vars)
            ))
            rand = Random(r_seed)
            test_insts: list[dict[str, str]] = []
            q_vars: list[str] = []

            for test_sample in test_samples:
                q_var = rand.choice(ov_vars)
                q_vars.append(q_var)
                evid: list[str] = rand.sample(
                    [n for n in ref_bn.nodes() if n != q_var], nr_evid_vars)
                test_insts.append({k: test_sample[k] for k in evid})

            res_single = benchmark_single(
                ref_model, trained_models, weights, test_insts, q_vars)
            res_single[_BENCHMARK_INDEX] = round(overlap, 1)
            d_f = pd.concat([d_f, res_single], ignore_index=True, copy=False)

    d_f.set_index(_BENCHMARK_INDEX, inplace=True)

    return d_f


def benchmark_single(
        ref_model: Model,
        trained_models: list[BayesianNetwork],
        weights: list[float],
        test_samples: list[dict[str, str]],
        query_vars: list[str]) -> pd.DataFrame:
    res_single: list[dict[str, float | str]] = []
    LOGGER.info("Getting reference results")
    ref_out = _get_inf_res(ref_model, test_samples, query_vars)

    for name, model in _yield_approaches(trained_models, weights):
        LOGGER.info("Benchmarking approach %s", name)
        row: dict[str, float | str] = {BENCHMARK_PIVOT_COL: name}
        pred_out = _get_inf_res(model, test_samples, query_vars)

        row["Brier"] = round(_calc_brier(ref_out.res_facts, pred_out.res_facts), 3)
        row["RelTotTime"] = round(pred_out.tot_time / ref_out.tot_time, 2)
        row["AvgCommVals"] = round(pred_out.avg_comm_vals)
        # row["AbsTotTime (ms)"] = round(pred_out.tot_time / 1e6)
        # row["StructureF1"] = round(sf1_score(ref_model, model), 3)
        # row["SHD"] = shd_score(ref_model, model)
        # row["EdgeCount"] = len(model.edges())
        res_single.append(row)

    return pd.DataFrame.from_records(res_single)


def _get_in_out_nodes(bayes_net: BayesianNetwork) -> tuple[list[str], list[str]]:
    # Island nodes are not included in neither the in, nor the out list.
    in_nodes: list[str] = []
    out_nodes: list[str] = []
    edges_to, edges_from = [
        set(cast(list[str], edge_list)) for edge_list in zip(*bayes_net.edges())]

    for node in cast(Collection[str], bayes_net.nodes()):
        if node not in edges_to and node in edges_from:
            out_nodes.append(node)
        elif node not in edges_from and node in edges_to:
            in_nodes.append(node)

    return (in_nodes, out_nodes)


def _split_vars(
        bayes_net: BayesianNetwork,
        nr_splits: int,
        overlap_proportion: float,
        connected: bool,
        seed: int | None = None) -> tuple[list[list[str]], list[str]]:
    rand = Random(seed)
    nr_overlaps = round(overlap_proportion * len(bayes_net))

    if connected:
        shuffled_edges = cast(list[tuple[str, str]], list(bayes_net.edges()))
        rand.shuffle(shuffled_edges)
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
        splits, ov_vars = _overlap_communities(communities, nr_overlaps, shuffled_edges)
    else:
        shuffled_nodes: list[str] = list(bayes_net.nodes())
        rand.shuffle(shuffled_nodes)
        ov_vars: set[str] = set(rand.sample(shuffled_nodes, nr_overlaps))
        splits: list[set[str]] = [
            ov_vars.union(x) for x in np.array_split(shuffled_nodes, nr_splits)]

    return [sorted(split) for split in splits], sorted(ov_vars)


@_memory.cache
def sample(bayes_net: BayesianNetwork, size: int, seed: int | None = None) -> pd.DataFrame:
    return BayesianModelSampling(bayes_net).forward_sample(
        size=size, seed=seed, show_progress=False, n_jobs=N_PARALLEL
    )


@_memory.cache
def _train_model(samples: pd.DataFrame, max_nr_parents: int, states: dict[str, list[str]]) -> \
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


def _overlap_communities(
        community_sets:  list[set[str]],
        nr_overlaps: int,
        shuffled_edges: list[tuple[str, str]]) -> tuple[list[set[str]], set[str]]:
    node_to_community: dict[str, int] = {}
    ov_nodes: set[str] = set()
    overlapped_communities: set[int] = set()
    remaining_edges: list[tuple[str, str]] = []

    for i, community in enumerate(community_sets):
        node_to_community.update({node: i for node in community})

    for node_out, node_inc in shuffled_edges:
        if len(ov_nodes) >= nr_overlaps:
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
            ov_nodes.add(node_out)
            ov_nodes.add(node_inc)
            overlapped_communities.update(edge_nodes_communities)
            community_sets[community_node_out].add(node_inc)
            community_sets[community_node_inc].add(node_out)

    for node_out, node_inc in remaining_edges:
        if len(ov_nodes) >= nr_overlaps:
            break

        ov_nodes.add(node_out)
        ov_nodes.add(node_inc)
        community_sets[node_to_community[node_out]].add(node_inc)
        community_sets[node_to_community[node_inc]].add(node_out)

    return community_sets, ov_nodes


def _get_inf_res(
        source: Model,
        samples: list[dict[str, str]],
        q_vars: list[str]) -> TestOut:
    facts: list[DiscreteFactor] = []
    tot_time = 0.0
    sum_comm_vals = 0

    for i, row in tqdm(enumerate(samples)):
        start = perf_counter_ns()
        query = source.query([q_vars[i]], row)

        tot_time += perf_counter_ns() - start
        facts.append(query)
        sum_comm_vals += source.last_nr_comm_vals

    return TestOut(facts, tot_time, sum_comm_vals / len(samples))


def _calc_brier(
        ref_facts: list[DiscreteFactor],
        pred_facts: list[DiscreteFactor]) -> float:
    acc = 0.0

    for i, ref_fact in enumerate(ref_facts):
        for state_comb in product(*ref_fact.state_names.values()):
            name_to_state = dict(zip(ref_fact.state_names.keys(), state_comb))
            ref_val = cast(float, ref_fact.get_value(**name_to_state))
            try:
                pred_val = cast(float, pred_facts[i].get_value(**name_to_state))
            except ValueError:
                pred_val = 0.0
            acc += (ref_val - pred_val) ** 2

    return float(acc / len(ref_facts))


def print_bn(bayes_net: BayesianNetwork, struct_only=False) -> None:
    if not struct_only:
        for cpd in bayes_net.get_cpds() or []:
            print(cpd)
            print()
    bayes_net.to_daft().render()
