from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from pgmpy.factors.discrete import DiscreteFactor


class Model(ABC):
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


def var_elim(targets: list[str], evidence: dict[str, str], facts: list[DiscreteFactor],
             nodes: Collection[str]) -> DiscreteFactor:
    node_to_facts: dict[str, list[DiscreteFactor]] = defaultdict(list)

    for fact in facts:
        for node in cast(list[str], fact.variables):
            node_to_facts[node].append(fact)

    for node in sorted(n for n in nodes if n not in evidence and n not in targets):
        rel_facts = node_to_facts[node].copy()

        facts = [factor for factor in facts if factor not in rel_facts]

        prod_rel_facts = DiscreteFactor([], [], [1])

        for rel_fact in rel_facts:
            prod_rel_facts.product(rel_fact, inplace=True)

        prod_rel_facts.marginalize([node], inplace=True)

        for node, node_facts in node_to_facts.items():
            node_to_facts[node] = [fact for fact in node_facts if fact not in rel_facts]

        for prf_var in cast(list[str], prod_rel_facts.variables):
            node_to_facts[prf_var].append(prod_rel_facts)

        facts.append(prod_rel_facts)

    prod_facts = DiscreteFactor([], [], [1])

    for fact in facts:
        prod_facts.product(fact, inplace=True)

    prod_facts.normalize(inplace=True)

    return prod_facts
