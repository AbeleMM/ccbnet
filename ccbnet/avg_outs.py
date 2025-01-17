from collections import defaultdict
from enum import Enum, auto

import networkx as nx
from disc_fact import DiscFactCfg
from model import Model
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork
from single_net import SingleNet
from var_elim_heurs import VarElimHeur


class MeanType(Enum):
    ARITH = auto()
    GEO = auto()


class AvgOuts(Model):
    def __init__(
            self, bayes_nets: list[BayesianNetwork], mean_type: MeanType,
            dfc: DiscFactCfg, veh: VarElimHeur) -> None:
        super().__init__(dfc, veh)
        self.nets = [SingleNet.from_bn(bn, False, dfc, veh) for bn in bayes_nets]
        self.mean_type = mean_type

    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        node_to_avg_fact: dict[str, DiscreteFactor] = {}
        node_to_nr_facts: defaultdict[str, int] = defaultdict(int)
        self.last_nr_comm_vals = 0

        for net in self.nets:
            node_to_fact = net.disjoint_query([t for t in targets if t in net.nodes()], evidence)
            for node, node_fact in node_to_fact.items():
                if node in node_to_avg_fact:
                    match self.mean_type:
                        case MeanType.ARITH:
                            node_to_avg_fact[node].values += node_fact.values
                        case MeanType.GEO:
                            node_to_avg_fact[node].values *= node_fact.values
                else:
                    node_to_avg_fact[node] = node_fact
                node_to_nr_facts[node] += 1
                self.last_nr_comm_vals += node_fact.values.size

        res = DiscreteFactor([], [], [1])

        for node, avg_fact in node_to_avg_fact.items():
            match self.mean_type:
                case MeanType.ARITH:
                    avg_fact.values /= node_to_nr_facts[node]
                case MeanType.GEO:
                    avg_fact.values **= 1 / node_to_nr_facts[node]
            avg_fact.normalize(inplace=True)
            res.product(avg_fact, inplace=True)

        res.normalize(inplace=True)

        return res

    def as_dig(self) -> nx.DiGraph:
        dig = nx.DiGraph()

        for bayes_net in self.nets:
            dig.add_edges_from(bayes_net.edges())

        return dig

    def update_dfc(self, dfc: DiscFactCfg) -> None:
        for net in self.nets:
            net.update_dfc(dfc)

    def update_veh(self, veh: VarElimHeur) -> None:
        for net in self.nets:
            net.update_veh(veh)
