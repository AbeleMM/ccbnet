import itertools
from collections import defaultdict
from typing import cast

import numpy as np
import numpy.typing as npt
from pgmpy.factors.discrete.CPD import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork


class Client:
    def __init__(self, identifier: int, local_bn: BayesianNetwork) -> None:
        self.identifier = identifier
        self.local_bn = local_bn
        self.combined_bn = BayesianNetwork()
        self.combined_bn.add_nodes_from(self.local_bn.nodes)
        self.clients: list[Client] = []
        self.node_to_neighbors: dict[str, list[Client]] = {}
        self.ov_nodes: set[str] = set()

    def add_clients(self, clients: list['Client']) -> None:
        self.clients = sorted(
            [client for client in clients if client.identifier != self.identifier],
            key=lambda x: x.identifier
        )

    def combine(self) -> None:
        self.find_overlaps()
        for client in self.clients:
            client.find_overlaps()

        self.solve_overlaps()
        for client in self.clients:
            client.solve_overlaps()

    def find_overlaps(self) -> None:
        node_to_neighbors_set: defaultdict[str, set[Client]] = defaultdict(set)

        for client in self.clients:
            intersect = set(self.local_bn.nodes).intersection(client.local_bn.nodes)
            for node in intersect:
                node_to_neighbors_set[node].add(client)

        self.node_to_neighbors = {
            node: sorted(neighbors, key=lambda x: x.identifier)
            for node, neighbors in node_to_neighbors_set.items()
        }
        self.ov_nodes = set(node_to_neighbors_set)

        for node in self.local_bn.nodes():
            if node not in self.node_to_neighbors:
                cpd = cast(TabularCPD, self.local_bn.get_cpds(node)).copy()
                for parent in cast(list[str], self.local_bn.get_parents(node)):
                    self.combined_bn.add_edge(parent, node)
                self.combined_bn.add_cpds(cpd)

    def solve_overlaps(self) -> None:
        for node, neighbors in self.node_to_neighbors.items():
            third_party = next(
                (client for client in self.clients if client not in neighbors),
                Client(-1, BayesianNetwork())
            )
            parents_union = third_party.get_parents_union(node, [self, *neighbors])
            self.add_parents(node, parents_union)

            for client in neighbors:
                client.add_parents(node, parents_union)

            cpd = third_party.get_combined_cpd(node, parents_union, [self, *neighbors])
            self.combined_bn.add_cpds(cpd)

            for client in neighbors:
                client.mark_overlap_solved(node)

    def get_parents_union(self, node: str, clients: list['Client']) -> list[str]:
        return sorted(set().union(*[client.local_bn.get_parents(node) for client in clients]))

    def add_parents(self, node: str, parents: list[str]) -> None:
        for parent in parents:
            self.combined_bn.add_edge(parent, node)

    def get_combined_cpd(
            self, node: str, parents_union: list[str], clients: list['Client']) -> TabularCPD:
        node_to_states = self.get_node_to_states([node, *parents_union], clients)
        values = sum(
            client.get_expanded_values(node, node_to_states) for client in clients
        ) / len(clients)

        return TabularCPD(
            variable=node,
            variable_card=len(node_to_states[node]),
            values=values,
            evidence=parents_union,
            evidence_card=[len(node_to_states[p]) for p in parents_union],
            state_names=node_to_states
        )

    def get_node_to_states(
            self, nodes: list[str], clients: list['Client']) -> dict[str, list[str]]:
        node_to_val_dict: dict[str, dict[str, None]] = {node: {} for node in nodes}

        for client in clients:
            for node in nodes:
                if node not in client.local_bn.states:
                    continue
                node_to_val_dict[node].update(dict.fromkeys(client.local_bn.states[node], None))

        return {node: list(values_dict.keys()) for node, values_dict in node_to_val_dict.items()}

    def get_expanded_values(
            self, node: str, node_to_states: dict[str, list[str]]) -> npt.NDArray[np.float_]:
        factor = cast(TabularCPD, self.local_bn.get_cpds(node)).to_factor()
        values: list[list[float]] = []
        parent_to_states = node_to_states.copy()
        del parent_to_states[node]
        parent_states_combinations = list(itertools.product(*list(parent_to_states.values())))

        for node_state in node_to_states[node]:
            col: list[float] = []

            # TODO generalize for when a node or evidence state is not available
            for parent_states_combination in parent_states_combinations:
                get_value_dict = {
                    parent: parent_states_combination[i]
                    for i, parent in enumerate(parent_to_states)
                    if parent in factor.variables
                }
                get_value_dict[node] = node_state
                col.append(cast(float, factor.get_value(**get_value_dict)))
            values.append(col)

        return np.array(values)

    def mark_overlap_solved(self, node) -> None:
        del self.node_to_neighbors[node]

    def elimination_ask(
            self, query: list[str], evidence: dict[str, str], propagate=True) -> DiscreteFactor:
        no_merge_vars: set[str] = set()
        if propagate:
            factors = [client.elimination_ask(query, evidence, False) for client in self.clients]
            factors.append(self.elimination_ask(query, evidence, False))
        else:
            factors = [
                cast(
                    DiscreteFactor,
                    cpd.to_factor().reduce(
                        [(var, evidence[var]) for var in cpd.variables if var in evidence],
                        inplace=False,
                        show_warnings=False
                    )
                )
                for cpd in cast(list[TabularCPD], self.combined_bn.get_cpds())
            ]
            no_merge_vars = set(self.ov_nodes)

        node_to_factors: dict[str, set[DiscreteFactor]] = defaultdict(set)
        query_set = set(query)

        for factor in factors:
            for variable in cast(list[str], factor.variables):
                node_to_factors[variable].add(factor)

        for var in node_to_factors:
            if var in evidence or var in query_set:
                continue

            relevant_factors = node_to_factors[var].copy()

            factors = [factor for factor in factors if factor not in relevant_factors]

            product_relevant_factors = DiscreteFactor([], [], [1])

            for relevant_factor in relevant_factors:
                product_relevant_factors.product(relevant_factor, inplace=True)

            if var not in no_merge_vars:
                product_relevant_factors.marginalize([var], inplace=True)

            for node_factors in node_to_factors.values():
                node_factors -= relevant_factors

            for prf_var in cast(list[str], product_relevant_factors.variables):
                node_to_factors[prf_var].add(product_relevant_factors)

            factors.append(product_relevant_factors)

        product_factors = DiscreteFactor([], [], [1])

        for factor in factors:
            product_factors.product(factor, inplace=True)

        product_factors.normalize(inplace=True)

        return product_factors

    def disjoint_elimination_ask(
            self, query: list[str], evidence: dict[str, str]) -> dict[str, DiscreteFactor]:
        factor = self.elimination_ask(query, evidence)

        return {
            var: cast(
                DiscreteFactor, factor.marginalize([q for q in query if q != var], inplace=False))
            for var in query
        }

    def map_elimination_ask(self, query: list[str], evidence: dict[str, str]) -> dict[str, str]:
        factor = self.elimination_ask(query, evidence)
        argmax = np.argmax(factor.values)
        assignment, *_ = cast(list[list[tuple[str, str]]], factor.assignment([argmax]))

        return dict(assignment)


def combine(bns: list[BayesianNetwork]) -> Client:
    clients = [Client(i, bn) for i, bn in enumerate(bns)]

    for client in clients:
        client.add_clients(clients)

    next(iter(clients)).combine()

    return min(clients, key=lambda x: x.identifier)
