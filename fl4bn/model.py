from abc import ABC, abstractmethod
from collections.abc import Collection
from operator import itemgetter
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from pgmpy.factors import factor_product
from pgmpy.factors.discrete import DiscreteFactor


class Model(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.last_nr_comm_vals = 0

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


def var_elim(factors: list[DiscreteFactor], nodes: set[str]) -> \
        DiscreteFactor:
    facts = factors.copy()
    remaining_nodes = set(nodes)

    while remaining_nodes:
        node_to_parties: dict[str, int] = {}
        for fact in facts:
            for var in fact.variables:
                if var not in remaining_nodes:
                    continue
                node_to_parties[var] = node_to_parties.get(var, 1) * fact.values.size
        node, _ = min(node_to_parties.items(), key=itemgetter(1, 0))
        remaining_nodes.remove(node)

        rel_facts: list[DiscreteFactor] = []
        new_facts: list[DiscreteFactor] = []

        for fact in facts:
            if node in fact.variables:
                rel_facts.append(fact)
            else:
                new_facts.append(fact)

        prod_rel_facts = cast(DiscreteFactor, factor_product(*rel_facts))
        prod_rel_facts.marginalize([node], inplace=True)
        facts = new_facts
        facts.append(prod_rel_facts)

    prod_facts = cast(DiscreteFactor, factor_product(*facts))
    prod_facts.normalize(inplace=True)

    return prod_facts
