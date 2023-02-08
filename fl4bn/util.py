import warnings
from collections import defaultdict
from random import Random
from typing import Collection, cast
from pathlib import Path

from combine import CombineMethod, combine_bns
from joblib import Memory
from pandas import DataFrame
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model

_memory = Memory(Path(__file__).parents[1] / "cache", verbose=0)


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


def split_vars(bayes_net: BayesianNetwork, nr_splits: int, r_seed: int | None) -> list[list[str]]:
    # Assumes nr. connected components in network <= nr. splits
    # Based on Kruskal's Algorithm
    node_parent: dict[str, str] = {}
    node_rank: dict[str, int] = {}
    shuffled_edges = cast(list[tuple[str, str]], list(bayes_net.edges()))
    Random(r_seed).shuffle(shuffled_edges)  # nosec
    nr_edges = 0

    for node in cast(Collection[str], bayes_net.nodes()):
        node_parent[node] = node
        node_rank[node] = 0

    while nr_edges < len(bayes_net.nodes()) - nr_splits:
        node_out, node_inc = shuffled_edges.pop()
        parent_node_out = _find_parent(node_out, node_parent)
        parent_node_inc = _find_parent(node_inc, node_parent)

        if parent_node_out != parent_node_inc:
            nr_edges += 1
            if node_rank[parent_node_out] < node_rank[parent_node_inc]:
                node_parent[parent_node_out] = node_parent[parent_node_inc]
            else:
                node_parent[parent_node_inc] = node_parent[parent_node_out]
                inc = node_rank[parent_node_out] == node_rank[parent_node_inc]
                node_rank[parent_node_out] += inc

    parent_to_nodes: dict[str, list[str]] = defaultdict(list)
    crossed_parents: set[str] = set()

    for node in node_parent:
        parent_to_nodes[_find_parent(node, node_parent)].append(node)

    for node_out, node_inc in shuffled_edges:
        parent_node_out = node_parent[node_out]
        parent_node_inc = node_parent[node_inc]
        if (
            parent_node_out != parent_node_inc and
            not set([parent_node_out, parent_node_inc]) <= crossed_parents
        ):
            crossed_parents.add(parent_node_out)
            crossed_parents.add(parent_node_inc)
            parent_to_nodes[parent_node_out].append(node_inc)
            parent_to_nodes[parent_node_inc].append(node_out)

    return list(parent_to_nodes.values())


@_memory.cache
def train_model(samples: DataFrame) -> BayesianNetwork:
    est = HillClimbSearch(data=samples, use_cache=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        estimated_dag = cast(
            DAG,
            est.estimate(
                scoring_method="k2score",
                max_indegree=len(samples.columns) // 2,
                max_iter=10**4,
                show_progress=False
            )
        )

    bayes_net = BayesianNetwork()
    bayes_net.add_nodes_from(samples.columns)
    bayes_net.add_edges_from(estimated_dag.edges())

    bayes_net.fit(
        samples,
        estimator=MaximumLikelihoodEstimator,
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


def benchmark(
        name_bn: str,
        nr_clients: int,
        test_counts: int,
        samples_per_node: int = 5000,
        r_seed: int | None = None) -> dict[str, dict[str, float]]:
    model = get_example_model(name_bn)
    sampling = BayesianModelSampling(model)
    clients_train_vars = split_vars(model, nr_clients, r_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        clients_train_samples = [
            sampling.forward_sample(
                size=samples_per_node * len(vars),
                seed=r_seed,
                show_progress=False
            )[vars]
            for vars in clients_train_vars
        ]
        test_samples = sampling.forward_sample(size=test_counts, seed=r_seed, show_progress=False)
    evidence_vars, query_vars = get_in_out_nodes(model)
    trained_models = [train_model(client_samples) for client_samples in clients_train_samples]
    res = {
        "Ground truth": calc_accuracy(test_samples, model, evidence_vars, query_vars)
    }
    for method in CombineMethod:
        res[method.value] = calc_accuracy(
            test_samples,
            # TODO switch to combine_bns_weighted (by amount of samples)
            combine_bns(trained_models, method),
            evidence_vars,
            query_vars
        )
    return res


def _find_parent(node: str, node_to_parent: dict[str, str]) -> str:
    parent = node
    while node_to_parent[parent] != parent:
        node_to_parent[parent] = node_to_parent[node_to_parent[parent]]
        parent = node_to_parent[parent]
    return parent
