from abc import ABC, abstractmethod
from collections import defaultdict, deque
from math import prod
from operator import itemgetter
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from disc_fact import DiscFact, DiscFactCfg
from pgmpy.factors.discrete import DiscreteFactor


class Model(ABC):
    @abstractmethod
    def __init__(self, dfc: DiscFactCfg) -> None:
        self.last_nr_comm_vals = 0
        self.node_to_nr_states: dict[str, int] = {}
        self.dfc = dfc
        self.base_fact = DiscFact([], [], 1, dfc=self.dfc)

    @abstractmethod
    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        ...

    @abstractmethod
    def as_dig(self) -> nx.DiGraph:
        ...

    def adj_mat(self, preserve_dir: bool) -> npt.NDArray[np.float_]:
        dig = self.as_dig()
        graph = dig if preserve_dir else dig.to_undirected()
        return nx.convert_matrix.to_numpy_array(graph, weight="")

    def disjoint_query(self, targets: list[str], evidence: dict[str, str]) -> \
            dict[str, DiscreteFactor]:
        factor = self.query(targets, evidence)
        return {
            var: cast(DiscreteFactor,
                      factor.marginalize([q for q in targets if q != var], inplace=False))
            for var in targets
        }

    def map_query(self, targets: list[str], evidence: dict[str, str]) -> dict[str, str]:
        factor = self.query(targets, evidence)
        argmax = np.argmax(factor.values)
        assignment, *_ = cast(list[list[tuple[str, str]]], factor.assignment([argmax]))
        return dict(assignment)


def var_elim(
        factors: list[DiscreteFactor], nodes: set[str], node_to_nr_states: dict[str, int],
        base_fact: DiscreteFactor) -> DiscreteFactor:
    remaining_nodes = set(nodes)
    facts = deque(factors)
    new_facts: deque[DiscreteFactor] = deque()

    while remaining_nodes:
        node_to_members: defaultdict[str, set[str]] = defaultdict(set)

        for fact in facts:
            for var in fact.variables:
                if var not in remaining_nodes:
                    continue

                members = node_to_members[var]
                members.update(fact.variables)
                members.remove(var)

        node, _ = min(
            ((n, prod(node_to_nr_states[v] for v in ms)) for n, ms in node_to_members.items()),
            key=itemgetter(1, 0))
        remaining_nodes.remove(node)
        prod_facts = base_fact.copy()

        while facts:
            fact = facts.popleft()
            if node in fact.variables:
                prod_facts.product(fact, inplace=True)
            else:
                new_facts.append(fact)

        prod_facts.marginalize([node], inplace=True)
        facts, new_facts = new_facts, facts
        facts.append(prod_facts)

    prod_facts = base_fact.copy()

    while facts:
        prod_facts.product(facts.popleft(), inplace=True)

    prod_facts.normalize(inplace=True)

    return prod_facts
