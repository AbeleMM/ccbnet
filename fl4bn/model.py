from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Collection
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from pgmpy.factors import factor_product
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


def var_elim(targets: list[str], evidence: dict[str, str], factors: list[DiscreteFactor],
             nodes: Collection[str]) -> DiscreteFactor:
    facts = factors.copy()

    for node in sorted(n for n in nodes if n not in evidence and n not in targets):
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
