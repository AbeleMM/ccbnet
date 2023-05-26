from pgmpy.factors.discrete import DiscreteFactor

try:
    import cupy as cp
    _USE_GPU = True
except ImportError:
    _USE_GPU = False


class DiscFactGPU(DiscreteFactor):
    def __init__(self, variables, cardinality, values, state_names=None):
        if state_names is None:
            state_names = {}
        super().__init__(variables, cardinality, values, state_names)
        self.state_names: dict
        self.no_to_name: dict
        self.name_to_no: dict
        if _USE_GPU:
            self.values = cp.asarray(self.values)

    @classmethod
    def from_disc_fact(cls, disc_fact: DiscreteFactor):
        return cls(
            disc_fact.variables, disc_fact.cardinality, disc_fact.values, disc_fact.state_names)

    def copy(self):
        copy = DiscreteFactor.__new__(self.__class__)
        copy.variables = self.variables.copy()
        copy.cardinality = self.cardinality.copy()
        copy.values = self.values.copy()
        copy.state_names = self.state_names.copy()
        copy.no_to_name = self.no_to_name.copy()
        copy.name_to_no = self.name_to_no.copy()
        return copy
