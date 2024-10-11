from dataclasses import dataclass, field
from typing import Iterable, cast

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD

try:
    import cupy as cp
    _USE_GPU = True
except ImportError:
    _USE_GPU = False


@dataclass(frozen=True)
class DiscFactCfg:
    """Data class specifying the config for a factor's values"""
    allow_gpu: bool = field(default=False)
    float_type: type = field(default=np.float_)


class DiscFact(DiscreteFactor):
    """DiscreteFactor subclass that abides by a given factor config"""
    # pylint: disable=super-init-not-called
    def __init__(
            self, variables, cardinality, values, state_names=None,
            dfc: DiscFactCfg | None = None):
        if state_names is None:
            state_names = {}
        if dfc is None:
            dfc = DiscFactCfg()
        self.state_names: dict
        self.no_to_name: dict
        self.name_to_no: dict

        if isinstance(variables, str):
            raise TypeError("Variables: Expected type list or array like, got string")

        self.values = values
        self.set_cfg(dfc)

        if len(cardinality) != len(variables):
            raise ValueError(
                "Number of elements in cardinality must be equal to number of variables"
            )

        card_prod = np.prod(cardinality)
        if self.values.size != card_prod:
            raise ValueError(f"Values array must be of size: {card_prod}")

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same")

        if not isinstance(state_names, dict):
            raise ValueError(
                f"state_names must be of type dict. Got {type(state_names)}."
            )

        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=np.int32)
        self.values.resize(self.cardinality)

        # Set the state names
        super(DiscreteFactor, self).store_state_names(
            variables, cardinality, state_names
        )

    @classmethod
    def from_cpd(cls, cpd: TabularCPD, dfc: DiscFactCfg | None = None):
        """Create a DiscFact from a CPD and config"""
        return cls(cpd.variables, cpd.cardinality, cpd.values, cpd.state_names, dfc)

    def copy(self):
        copy = DiscreteFactor.__new__(self.__class__)
        copy.variables = self.variables.copy()
        copy.cardinality = self.cardinality.copy()
        copy.values = self.values.copy()
        copy.state_names = self.state_names.copy()
        copy.no_to_name = self.no_to_name.copy()
        copy.name_to_no = self.name_to_no.copy()
        return copy

    def set_cfg(self, dfc: DiscFactCfg) -> None:
        """Update the config of the current factor"""
        if _USE_GPU and dfc.allow_gpu:
            self.values = cp.asarray(self.values, dtype=dfc.float_type)
        else:
            self.values = np.array(self.values, dtype=dfc.float_type)


def fact_prod(facts: Iterable[DiscreteFactor], dfc: DiscFactCfg) -> DiscreteFactor:
    """
    Alternate method for calculating product of a list of factors in one go.
    It tends to speed up GPU-accelerated inference, but slow down the CPU-only version.
    When >52 indices are used in the einsum expression, errors out with "too many subscripts".
    Could be updated to keep track of the subscript count and perform more einsum calls if needed.
    """
    var_to_states: dict[str, list[str]] = {
        var: states
        for fact in facts
        for var, states in cast(dict[str, list[str]], fact.state_names).items()
    }
    var_to_int = {var: index for index, var in enumerate(var_to_states)}
    args: list = [
        x for fact in facts for x in (fact.values, [var_to_int[var] for var in fact.variables])]
    args.append(range(len(var_to_states)))
    vals = np.einsum(*args)
    card = list(map(len, var_to_states.values()))
    return DiscFact(var_to_states, card, vals, var_to_states, dfc)
