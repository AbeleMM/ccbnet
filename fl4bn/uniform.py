import networkx as nx
from model import Model
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import BayesianNetwork


class Uniform(Model):
    def __init__(self, bns: list[BayesianNetwork]) -> None:
        self.states: dict[str, list[str]] = {}
        for bayes_net in bns:
            self.states.update(bayes_net.states)

    def query(self, targets: list[str], _: dict[str, str]) -> DiscreteFactor:
        tgt_cards: list[int] = []
        tgt_to_states: dict[str, list[str]] = {}
        tot_size = 1

        for target in targets:
            target_states = self.states[target]
            target_states_len = len(target_states)
            tgt_cards.append(target_states_len)
            tot_size *= target_states_len
            tgt_to_states[target] = target_states

        disc_fact = DiscreteFactor(
            targets,
            tgt_cards,
            [1] * tot_size,
            tgt_to_states
        )
        disc_fact.normalize(inplace=True)

        return disc_fact

    def as_dig(self) -> nx.DiGraph:
        return nx.DiGraph()
