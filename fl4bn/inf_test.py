from typing import cast

import inference as inf
import networkx as nx
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model
from util import get_in_out_nodes


def share_net(bayes_net: BayesianNetwork, nr_splits: int) -> list[BayesianNetwork]:
    shared_nets = [bayes_net.copy() for _ in range(nr_splits)]

    dfs_tree = nx.dfs_tree(bayes_net.to_undirected())
    communities: list[set[str]] = [
        set(community)
        for community in nx.algorithms.community.greedy_modularity_communities(
            dfs_tree,
            cutoff=nr_splits,
            best_n=nr_splits
        )
    ]
    # print(communities)

    for ind, shared_net in enumerate(shared_nets):
        shared_net_nodes = communities[ind]
        shared_net.remove_cpds(*[
            cpd
            for cpd in cast(list[TabularCPD], shared_net.get_cpds())
            if cpd.variable not in shared_net_nodes
        ])

    return shared_nets


def calc_acc(
        test_df: pd.DataFrame,
        source: VariableElimination | BayesianNetwork | list[BayesianNetwork],
        e_vars: list[str], q_vars: set[str]) -> dict[str, float]:
    accuracy = {v: 0 for v in q_vars}

    for i, row in test_df.iterrows():
        # print(i)
        evid = {v: row[v] for v in e_vars}
        if isinstance(source, VariableElimination):
            query = cast(
                dict[str, str],
                source.map_query(
                    variables=q_vars,
                    evidence=evid,
                    show_progress=False
                )
            )
        else:
            query = inf.map_elimination_ask(source, q_vars, evid)
        for variable, value in query.items():
            accuracy[variable] += row[variable] == value

    acc = {k: v / len(test_df) * 100 for (k, v) in accuracy.items()}
    return acc


def main() -> None:
    for data in ["asia"]:
        model = get_example_model(data)
        shared_models = share_net(model, 4)

        # query = {"xray"}
        # evidence = {"smoke": "no"}
        # model_ve = VariableElimination(model)
        # print(model_ve.query(query, evidence))
        # print(inf.elimination_ask(model, query, evidence))
        # # print("---")
        # # for cpd in model.get_cpds():
        # #     print(cpd)
        # # print("|||")
        # print(inf.elimination_ask(shared_models, query, evidence))

        sampling = BayesianModelSampling(model)
        test_samples = sampling.forward_sample(size=2000, seed=42, show_progress=False)
        evidence_vars, query_vars = get_in_out_nodes(model)
        query_vars = set(query_vars)
        print(data)
        acc_default = calc_acc(test_samples, VariableElimination(model), evidence_vars, query_vars)
        acc_own = calc_acc(test_samples, model, evidence_vars, query_vars)
        acc_shared = calc_acc(test_samples, shared_models, evidence_vars, query_vars)
        print("default", sum(acc_default.values()) / len(query_vars))
        print("own", sum(acc_own.values()) / len(query_vars))
        print("shared", sum(acc_shared.values()) / len(query_vars))


if __name__ == "__main__":
    main()
