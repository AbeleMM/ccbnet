from typing import cast

import networkx as nx
from model import Model, var_elim
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork


class ProdOuts(Model):
    def __init__(self, bayes_nets: list[BayesianNetwork]) -> None:
        self.bayes_nets = bayes_nets

    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        res = DiscreteFactor([], [], [1])

        for bayes_net in self.bayes_nets:
            facts = [
                cast(DiscreteFactor, cpd.to_factor().reduce(
                    [(var, evidence[var]) for var in cpd.variables if var in evidence],
                    inplace=False, show_warnings=True))
                for cpd in cast(list[TabularCPD], bayes_net.get_cpds())]

            res.product(
                var_elim(targets, evidence, facts, list(bayes_net.nodes()), set()), inplace=True)

        res.normalize(inplace=True)

        return res

    def as_dig(self) -> nx.DiGraph:
        dig = nx.DiGraph()

        for bayes_net in self.bayes_nets:
            dig.add_edges_from(bayes_net.edges())

        return dig
