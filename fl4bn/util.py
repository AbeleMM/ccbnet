import warnings
from collections import Counter, deque
from math import ceil
from pathlib import Path
from random import Random
from typing import Any, Collection, cast

import networkx as nx
import numpy as np
import pandas as pd
from combine import CombineMethod, combine_bns
from joblib import Memory
from pandas import DataFrame
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from sklearn.metrics import f1_score

_memory = Memory(Path(__file__).parents[1] / "cache", verbose=0)
_BENCHMARK_INDEX = "Overlap"
BENCHMARK_PIVOT_COL = "Method"


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
            for community in nx.algorithms.community.greedy_modularity_communities(
                dfs_tree,
                cutoff=nr_splits,
                best_n=nr_splits
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
    _overlap_communities(communities, nr_overlaps, deque(shuffled_edges))

    return [sorted(community) for community in communities]


@_memory.cache
def train_model(samples: DataFrame, max_nr_parents: int) -> BayesianNetwork:
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


def calc_accuracy(
        test_df: DataFrame, bayes_net: BayesianNetwork,
        e_vars: list[str], q_vars: list[str]) -> dict[str, float]:
    bn_infer = VariableElimination(bayes_net)
    accuracy = {v: 0 for v in q_vars}

    for _, row in test_df.iterrows():
        query = cast(
            dict[str, str],
            bn_infer.map_query(
                variables=q_vars,
                evidence={v: row[v] for v in e_vars},
                show_progress=False
            )
        )
        for variable, value in query.items():
            accuracy[variable] += row[variable] == value

    accuracy = {k: round(v / len(test_df) * 100, 3) for (k, v) in accuracy.items()}
    return accuracy


def benchmark_single(
        ref_model: BayesianNetwork,
        trained_models: list[BayesianNetwork],
        test_samples: DataFrame,
        evidence_vars: list[str],
        query_vars: list[str]) -> DataFrame:
    ref_accuracy = calc_accuracy(test_samples, ref_model, evidence_vars, query_vars)
    res_single: list[dict[str, Any]] = []

    for method in CombineMethod:
        # TODO switch to combine_bns_weighted (by amount of samples)
        bayes_net = combine_bns(trained_models, method)
        row: dict[str, Any] = {BENCHMARK_PIVOT_COL: method.value}

        acc = 0.0

        for k, val in calc_accuracy(test_samples, bayes_net, evidence_vars, query_vars).items():
            acc += val / ref_accuracy[k]

        acc /= len(ref_accuracy)
        row["Relative Accuracy"] = round(acc, 3)

        row["Structure F1"] = round(sf1_score(ref_model, bayes_net), 3)

        row["SHD"] = round(shd_score(ref_model, bayes_net), 3)

        res_single.append(row)

    return pd.DataFrame.from_records(res_single)


def benchmark_multi(
        ref_model: BayesianNetwork,
        nr_clients: int,
        test_counts: int = 2000,
        samples_factor: int = 50000,
        r_seed: int | None = None) -> DataFrame:
    sampling = BayesianModelSampling(ref_model)
    samples_per_client = samples_factor * len(ref_model.nodes()) // nr_clients
    res_multi = DataFrame()

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
        test_samples = sampling.forward_sample(size=test_counts, seed=r_seed, show_progress=False)
        evidence_vars, query_vars = get_in_out_nodes(ref_model)
        max_in_deg = Counter(e[1] for e in ref_model.edges()).most_common(1)[0][1]

        for overlap in np.arange(0.0, 0.6, 0.1):
            clients_train_vars = split_vars(ref_model, nr_clients, overlap, seed=r_seed)

            trained_models = [
                train_model(clients_train_samples[i][train_vars], max_in_deg)
                for i, train_vars in enumerate(clients_train_vars)
            ]

            res_single = benchmark_single(
                ref_model, trained_models, test_samples, evidence_vars, query_vars
            )
            res_single[_BENCHMARK_INDEX] = overlap

            res_multi = pd.concat([res_multi, res_single], ignore_index=True, copy=False)

    res_multi.set_index(_BENCHMARK_INDEX, inplace=True)

    return res_multi


def _overlap_communities(
        communities:  list[set[str]],
        nr_overlaps: int,
        shuffled_edges: deque[tuple[str, str]]) -> None:
    node_to_community: dict[str, int] = {}
    overlap_nodes: set[str] = set()
    overlapped_communities: set[int] = set()

    for i, community in enumerate(communities):
        node_to_community.update({node: i for node in community})

    while (
        shuffled_edges and
        (len(overlapped_communities) < len(communities) or len(overlap_nodes) < nr_overlaps)
    ):
        node_out, node_inc = shuffled_edges.popleft()
        community_node_out = node_to_community[node_out]
        community_node_inc = node_to_community[node_inc]

        if community_node_out == community_node_inc:
            continue

        edge_nodes_communities = set([community_node_out, community_node_inc])

        if (
            len(overlapped_communities) < len(communities) and
            edge_nodes_communities <= overlapped_communities
        ):
            shuffled_edges.append((node_out, node_inc))
        else:
            overlap_nodes.add(node_out)
            overlap_nodes.add(node_inc)
            overlapped_communities.update(edge_nodes_communities)
            communities[community_node_out].add(node_inc)
            communities[community_node_inc].add(node_out)


def _bn_to_adj_mat(model: BayesianNetwork, preserve_dir) -> np.ndarray:
    model_transformed = model if preserve_dir else model.to_undirected()
    return nx.to_numpy_array(model_transformed, nodelist=model.nodes(), weight="")


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
