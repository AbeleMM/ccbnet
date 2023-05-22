from typing import cast

import networkx as nx
from model import Model, var_elim
from pgmpy.base import DAG
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork


class SingleNet(Model, BayesianNetwork):
    def __init__(self, allow_loops: bool) -> None:
        super().__init__()
        self.allow_loops = allow_loops

    @classmethod
    def from_bn(cls, bayes_net: BayesianNetwork, allow_loops=False) -> "SingleNet":
        single_net = cls(allow_loops)
        single_net.add_nodes_from(bayes_net.nodes())
        single_net.add_edges_from(bayes_net.edges())
        single_net.add_cpds(*cast(list[TabularCPD], bayes_net.get_cpds()))
        return single_net

    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        facts = [
            cast(DiscreteFactor, cpd.to_factor().reduce(
                [(var, evidence[var]) for var in cpd.variables if var in evidence],
                inplace=False, show_warnings=True))
            for cpd in cast(list[TabularCPD], self.get_cpds())]

        return var_elim(targets, evidence, facts, list(self.nodes()), set())

    def as_dig(self) -> nx.DiGraph:
        return self

    def add_edge(self, u: str, v: str, **kwargs) -> None:
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if not self.allow_loops and u in self.nodes() and v in self.nodes() and \
                nx.has_path(self, v, u):
            raise ValueError(("Loops are not allowed. Adding the edge from"
                              f"({u} -> {v}) forms a loop."))
        super(DAG, self).add_edge(u, v, **kwargs)
