from typing import cast

import networkx as nx
from disc_fact import DiscFact, DiscFactCfg
from model import Model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork
from var_elim_heurs import VarElimHeur


class SingleNet(Model, BayesianNetwork):
    def __init__(self, allow_loops: bool, dfc: DiscFactCfg, veh: VarElimHeur) -> None:
        Model.__init__(self, dfc, veh)
        BayesianNetwork.__init__(self)
        self.allow_loops = allow_loops
        self.node_to_fact: dict[str, DiscFact] = {}

    @classmethod
    def from_bn(
            cls, bayes_net: BayesianNetwork, allow_loops: bool,
            dfc: DiscFactCfg, veh: VarElimHeur) -> "SingleNet":
        single_net = cls(allow_loops, dfc, veh)
        single_net.add_nodes_from(bayes_net.nodes())
        single_net.add_edges_from(bayes_net.edges())
        single_net.add_cpds(*cast(list[TabularCPD], bayes_net.get_cpds()))
        return single_net

    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        facts = [
            cast(DiscreteFactor, fact.reduce(
                [(var, evidence[var]) for var in fact.variables if var in evidence],
                inplace=False, show_warnings=True))
            for fact in self.node_to_fact.values()]
        discard: set[str] = set().union(targets, evidence)
        nodes: set[str] = set(n for n in self.nodes() if n not in discard)

        return self.var_elim(facts, nodes, self.node_to_nr_states)

    def as_dig(self) -> nx.DiGraph:
        return self

    def add_edge(self, u: str, v: str, **kwargs) -> None:
        if u == v:
            raise ValueError("Self loops are not allowed.")
        if not self.allow_loops and u in self.nodes() and v in self.nodes() and \
                nx.has_path(self, v, u):
            raise ValueError(("Loops are not allowed. Adding the edge from"
                              f"({u} -> {v}) forms a loop."))
        super(BayesianNetwork, self).add_edge(u, v, **kwargs)

    def add_cpds(self, *cpds: TabularCPD) -> None:
        super().add_cpds(*cpds)
        for cpd in cpds:
            self.node_to_fact[cpd.variable] = DiscFact.from_cpd(cpd, self.dfc)
            self.node_to_nr_states.update((n, len(s)) for n, s in cpd.state_names.items())

    def update_dfc(self, dfc: DiscFactCfg) -> None:
        super().update_dfc(dfc)
        for fact in self.node_to_fact.values():
            fact.set_cfg(dfc)
