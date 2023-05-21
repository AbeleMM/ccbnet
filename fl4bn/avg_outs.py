from collections import defaultdict

import networkx as nx
from model import Model
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork
from single_net import SingleNet


class AvgOuts(Model):
    def __init__(self, bayes_nets: list[BayesianNetwork]) -> None:
        self.nets = [SingleNet.from_bn(bn, False) for bn in bayes_nets]

    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        node_to_avg_fact: dict[str, DiscreteFactor] = {}
        node_to_nr_facts: defaultdict[str, int] = defaultdict(int)

        for net in self.nets:
            node_to_fact = net.disjoint_query(sorted(set(net.nodes()) & set(targets)), evidence)
            for node, node_fact in node_to_fact.items():
                if node in node_to_avg_fact:
                    node_to_avg_fact[node].values += node_fact.values
                else:
                    node_to_avg_fact[node] = node_fact
                node_to_nr_facts[node] += 1

        res = DiscreteFactor([], [], [1])

        for node, avg_fact in node_to_avg_fact.items():
            avg_fact.values /= node_to_nr_facts[node]
            avg_fact.normalize(inplace=True)
            res.product(avg_fact, inplace=True)

        res.normalize(inplace=True)

        return res

    def as_dig(self) -> nx.DiGraph:
        dig = nx.DiGraph()

        for bayes_net in self.nets:
            dig.add_edges_from(bayes_net.edges())

        return dig
