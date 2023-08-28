from abc import ABC, abstractmethod
from collections import defaultdict
from math import prod
from operator import itemgetter
from typing import Iterable

from pgmpy.factors.discrete import DiscreteFactor


class VarElimHeur(ABC):
    """Abstract base class for variable elimination ordering heuristic"""
    @staticmethod
    @abstractmethod
    def find_var(
            facts: Iterable[DiscreteFactor],
            remaining_nodes: set[str],
            node_to_nr_states: dict[str, int]) -> str:
        """Find the next variable to eliminate"""
        ...


class NoVEH(VarElimHeur):
    """Simply eliminate the next variable in the order it is presented"""
    @staticmethod
    def find_var(
            facts: Iterable[DiscreteFactor],
            remaining_nodes: set[str],
            _node_to_nr_states: dict[str, int]) -> str:
        for fact in facts:
            for var in fact.variables:
                if var in remaining_nodes:
                    return var

        return ""


class MinNeighborsVEH(VarElimHeur):
    """Eliminate the variable present in the fewest factors"""
    @staticmethod
    def find_var(
            facts: Iterable[DiscreteFactor],
            remaining_nodes: set[str],
            _node_to_nr_states: dict[str, int]) -> str:
        node_to_nr_facts: defaultdict[str, int] = defaultdict(int)

        for fact in facts:
            for var in fact.variables:
                if var in remaining_nodes:
                    node_to_nr_facts[var] += 1

        node, _nr = min(node_to_nr_facts.items(), key=itemgetter(1, 0))

        return node


class MinWeightVEH(VarElimHeur):
    """Eliminate the variable yielding the smallest factor product"""
    @staticmethod
    def find_var(
            facts: Iterable[DiscreteFactor],
            remaining_nodes: set[str],
            node_to_nr_states: dict[str, int]) -> str:
        node_to_members: defaultdict[str, set[str]] = defaultdict(set)

        for fact in facts:
            for var in fact.variables:
                if var in remaining_nodes:
                    node_to_members[var].update(fact.variables)

        node, _nr = min(
            ((n, prod(node_to_nr_states[v] for v in ms)) for n, ms in node_to_members.items()),
            key=itemgetter(1, 0))

        return node
