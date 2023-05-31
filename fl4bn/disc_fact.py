from typing import NamedTuple

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD

try:
    import cupy as cp
    _USE_GPU = True
except ImportError:
    _USE_GPU = False


class DiscFactCfg(NamedTuple):
    allow_gpu = False
    float_type = np.float_


class DiscFact(DiscreteFactor):
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

        if _USE_GPU and dfc.allow_gpu:
            values = cp.asarray(values, dtype=dfc.float_type)
        else:
            values = np.array(values, dtype=dfc.float_type)

        if len(cardinality) != len(variables):
            raise ValueError(
                "Number of elements in cardinality must be equal to number of variables"
            )

        card_prod = np.product(cardinality)
        if values.size != card_prod:
            raise ValueError(f"Values array must be of size: {card_prod}")

        if len(set(variables)) != len(variables):
            raise ValueError("Variable names cannot be same")

        if not isinstance(state_names, dict):
            raise ValueError(
                f"state_names must be of type dict. Got {type(state_names)}."
            )

        self.variables = list(variables)
        self.cardinality = np.array(cardinality, dtype=np.int32)
        self.values = values.reshape(self.cardinality)

        # Set the state names
        super(DiscreteFactor, self).store_state_names(
            variables, cardinality, state_names
        )

    @classmethod
    def from_cpd(cls, cpd: TabularCPD, dfc: DiscFactCfg | None = None):
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
