from abc import ABC, abstractmethod
from collections import deque
from typing import Literal, cast, overload

import networkx as nx
import numpy as np
import numpy.typing as npt
from disc_fact import DiscFact, DiscFactCfg
from pgmpy.factors.discrete import DiscreteFactor
from var_elim_heurs import VarElimHeur


class Model(ABC):
    """Abstract base class providing a common interface for BN-based methods"""
    @abstractmethod
    def __init__(self, dfc: DiscFactCfg, veh: VarElimHeur) -> None:
        self.last_nr_comm_vals = 0
        self.node_to_nr_states: dict[str, int] = {}
        self.dfc = dfc
        self.base_fact = DiscFact([], [], 1, dfc=self.dfc)
        self.veh = veh

    @abstractmethod
    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        """Run inference on the model"""
        ...

    @abstractmethod
    def as_dig(self) -> nx.DiGraph:
        """Get the directed graph backing the model"""
        ...

    def adj_mat(self, preserve_dir: bool) -> npt.NDArray[np.float_]:
        """Get adjacency matrix of (un)directed version of the graph backing the model"""
        dig = self.as_dig()
        graph = dig if preserve_dir else dig.to_undirected()
        return nx.convert_matrix.to_numpy_array(graph, weight="")

    def disjoint_query(self, targets: list[str], evidence: dict[str, str]) -> \
            dict[str, DiscreteFactor]:
        """Inference query that returns a separate factor for each target"""
        factor = self.query(targets, evidence)
        return {
            var: cast(DiscreteFactor,
                      factor.marginalize([q for q in targets if q != var], inplace=False))
            for var in targets
        }

    def map_query(self, targets: list[str], evidence: dict[str, str]) -> dict[str, str]:
        """Maximum a posteriori inference query"""
        factor = self.query(targets, evidence)
        argmax = np.argmax(factor.values)
        assignment, *_ = cast(list[list[tuple[str, str]]], factor.assignment([argmax]))
        return dict(assignment)

    @overload
    def var_elim(
            self, factors: list[DiscreteFactor],
            nodes: set[str], node_to_nr_states: dict[str, int],
            multi: Literal[False] = ...) -> DiscreteFactor:
        ...


    @overload
    def var_elim(
            self, factors: list[DiscreteFactor],
            nodes: set[str], node_to_nr_states: dict[str, int],
            multi: Literal[True] = ...) -> list[DiscreteFactor]:
        ...


    def var_elim(
            self, factors: list[DiscreteFactor],
            nodes: set[str], node_to_nr_states: dict[str, int],
            multi=False) -> DiscreteFactor | list[DiscreteFactor]:
        """Base variable elimination inference implementation"""
        remaining_nodes = set(nodes)
        facts = deque(factors)
        new_facts: deque[DiscreteFactor] = deque()

        while remaining_nodes:
            node = self.veh.find_var(facts, remaining_nodes, node_to_nr_states)
            remaining_nodes.remove(node)
            prod_facts = self.base_fact.copy()

            while facts:
                fact = facts.popleft()
                if node in fact.variables:
                    prod_facts.product(fact, inplace=True)
                else:
                    new_facts.append(fact)

            prod_facts.marginalize([node], inplace=True)
            facts, new_facts = new_facts, facts
            facts.append(prod_facts)

        prod_facts = self.base_fact.copy()

        if multi:
            return list(facts)

        while facts:
            prod_facts.product(facts.popleft(), inplace=True)

        prod_facts.normalize(inplace=True)

        return prod_facts

    def update_dfc(self, dfc: DiscFactCfg) -> None:
        """Update the config used for factor values"""
        self.dfc = dfc
        self.base_fact.set_cfg(dfc)

    def update_veh(self, veh: VarElimHeur) -> None:
        """Update the variable elimination ordering heuristic used for inference"""
        self.veh = veh
