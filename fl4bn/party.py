import itertools
from collections import defaultdict
from collections.abc import Collection
from operator import attrgetter
from typing import cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import private_set_intersection.python as psi
import tenseal as ts
from disc_fact import DiscFact, DiscFactCfg
from model import Model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork
from var_elim_heurs import VarElimHeur

REVEAL_INTERSECTION = True
FPR = 0.0
DS = psi.DataStructure.RAW
PMD_TO_MAX_CM_BITS = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
HE_DEC_BITS = 40
MIN_VAL = 0.1


class Party(Model):
    def __init__(
            self, identifier: int, local_bn: BayesianNetwork, weight: float, split_ov: bool,
            dfc: DiscFactCfg, veh: VarElimHeur) -> None:
        super().__init__(dfc, veh)
        self.identifier = identifier
        self.local_bn = local_bn
        self.weight = weight
        self.split_ov = split_ov
        self.node_to_cpd: dict[str, TabularCPD] = {
            cpd.variable: cpd for cpd in cast(list[TabularCPD], local_bn.get_cpds())
        }
        self.node_to_fact = {
            node: DiscFact.from_cpd(cpd, self.dfc) for node, cpd in self.node_to_cpd.items()}
        self.node_to_nr_states = {n: len(s) for n, s in local_bn.states.items()}
        self.peers: list[Party] = []
        self.node_to_neighbors: dict[str, list[Party]] = defaultdict(list)
        self.solved_overlaps: set[str] = set()
        self.rand_gen = np.random.default_rng(seed=self.identifier)
        self.no_marg_nodes: list[str] = []
        self.tmp_vals: npt.NDArray[np.float_]

    def query(self, targets: list[str], evidence: dict[str, str]) -> DiscreteFactor:
        facts: list[DiscreteFactor] = []
        self.last_nr_comm_vals = 0

        for party in cast(Party, self), *self.peers:
            party_facts = [
                cast(
                    DiscreteFactor,
                    fact.reduce(
                        [(var, evidence[var]) for var in fact.variables if var in evidence],
                        inplace=False,
                        show_warnings=False
                    )
                )
                for fact in party.node_to_fact.values()
            ]
            discard: set[str] = set().union(targets, party.no_marg_nodes, evidence)
            nodes = set(n for n in party.node_to_cpd if n not in discard)
            res_fact = self.var_elim(party_facts, nodes, party.node_to_nr_states)
            facts.append(res_fact)
            self.last_nr_comm_vals += res_fact.values.size

        nodes: set[str] = set()
        node_to_nr_states: dict[str, int] = {}

        for fact in facts:
            nodes.update(fact.variables)
            node_to_nr_states.update((n, len(s)) for n, s in fact.state_names.items())

        nodes.difference_update(targets)

        return self.var_elim(facts, nodes, node_to_nr_states)

    def as_dig(self) -> nx.DiGraph:
        dig = nx.DiGraph()

        for party in [cast(Party, self), *self.peers]:
            dig.add_edges_from(
                [(p, n) for n, f in party.node_to_cpd.items() for p in f.get_evidence()])

        return dig

    def add_peers(self, parties: list['Party']) -> None:
        self.peers.extend(p for p in parties if p.identifier != self.identifier)
        self.peers = sorted(set(self.peers), key=attrgetter("identifier"))

    def combine(self) -> None:
        for party in cast(Party, self), *self.peers:
            party.find_overlaps()

        for party in cast(Party, self), *self.peers:
            party.solve_overlaps()

    def find_overlaps(self) -> None:
        own_nodes_list = list(self.local_bn.nodes)
        for party in self.peers:
            if party.identifier < self.identifier:
                continue

            party_nodes_list = list(party.local_bn.nodes)
            psi_c = psi.client.CreateWithNewKey(REVEAL_INTERSECTION)
            psi_s = psi.server.CreateWithNewKey(REVEAL_INTERSECTION)
            setup = psi.ServerSetup()
            setup.ParseFromString(
                psi_s.CreateSetupMessage(
                    FPR, len(party_nodes_list), own_nodes_list, DS
                ).SerializeToString()
            )
            request = psi.Request()
            request.ParseFromString(
                psi_c.CreateRequest(party_nodes_list).SerializeToString()
            )
            response = psi.Response()
            response.ParseFromString(psi_s.ProcessRequest(request).SerializeToString())
            intersect: list[str] = [
                party_nodes_list[i] for i in psi_c.GetIntersection(setup, response)
            ]
            self.update_overlaps(party, intersect)
            party.update_overlaps(self, intersect)

        self.node_to_neighbors = {
            node: sorted(set(neighbors), key=lambda x: x.identifier)
            for node, neighbors in self.node_to_neighbors.items()
        }

    def update_overlaps(self, party: 'Party', intersect: list[str]) -> None:
        for node in intersect:
            self.node_to_neighbors[node].append(party)

    def solve_overlaps(self) -> None:
        for node in self.node_to_neighbors:
            if node in self.solved_overlaps:
                continue

            neighbors = self.node_to_neighbors[node]
            overlap_parties: list[Party] = [self, *neighbors]
            parents_union = self._get_parents_union(node, overlap_parties)
            nr_parties = len(overlap_parties)
            weight_sum = sum(p.weight for p in overlap_parties)
            node_to_states = self._get_node_to_states([node, *parents_union], overlap_parties)
            context = self._gen_context(nr_parties, len(node_to_states[node]))
            self.set_vals_ret_enc(node, weight_sum, node_to_states, context)
            enc_cols_parties = [
                party.set_vals_ret_enc(node, weight_sum, node_to_states, context)
                for party in neighbors
            ]
            column_sums: list[float] = self._calc_col_inner_prods(enc_cols_parties, nr_parties)

            party_unmixed_shares = [
                party.share_values(party.tmp_vals, nr_parties)
                for party in overlap_parties
            ]

            for i, shares in enumerate(map(list, zip(*party_unmixed_shares))):
                overlap_parties[i].tmp_vals = np.array([1.0])
                for share in shares:
                    overlap_parties[i].tmp_vals = overlap_parties[i].tmp_vals * share

            if self.split_ov:
                for party in overlap_parties:
                    cpd = party.get_combined_cpd(node, column_sums, parents_union, node_to_states)
                    party.mark_overlap_solved(node, node_to_states, cpd)
            else:
                cpd = self.get_combined_cpd(node, column_sums, parents_union, node_to_states)

                for party in neighbors:
                    cpd.values *= party.get_combined_cpd(
                        node, column_sums, parents_union, node_to_states).values
                    party.mark_overlap_solved(node, node_to_states, None)
                self.mark_overlap_solved(node, node_to_states, cpd)

        self.no_marg_nodes = sorted(set(self.no_marg_nodes))

    def _get_parents_union(self, node: str, parties: list['Party']) -> list[str]:
        return sorted(set().union(*[party.local_bn.get_parents(node) for party in parties]))

    def _get_node_to_states(
            self, nodes: list[str], parties: list['Party']) -> dict[str, list[str]]:
        node_to_val_dict: dict[str, dict[str, None]] = {node: {} for node in nodes}

        for party in parties:
            for node in nodes:
                if node not in party.local_bn.states:
                    continue
                node_to_val_dict[node].update(dict.fromkeys(party.local_bn.states[node], None))

        return {node: list(values_dict) for node, values_dict in node_to_val_dict.items()}

    def _gen_context(self, nr_parties: int, nr_states: int) -> ts.Context:
        aux = nr_states.bit_length()  # prec before dec point & inner/outer bits diff
        inner_bits = HE_DEC_BITS + aux
        outer_bits = inner_bits + aux
        cmb_sizes = [outer_bits, *([inner_bits] * nr_parties), outer_bits]
        cm_bits = sum(cmb_sizes)
        pm_degree = 0
        for pmd, max_cm_bits in PMD_TO_MAX_CM_BITS.items():
            if cm_bits < max_cm_bits:
                pm_degree = pmd
                break
        context = ts.context(ts.SCHEME_TYPE.CKKS,
                             poly_modulus_degree=pm_degree, coeff_mod_bit_sizes=cmb_sizes)
        context.generate_galois_keys()
        context.global_scale = 2**inner_bits
        return context

    def set_vals_ret_enc(
            self, node: str, weight_sum: float,
            node_to_states: dict[str, list[str]],
            context: ts.Context | None) -> list[ts.CKKSVector | npt.NDArray[np.float_]]:
        values = self._get_expanded_values(node, node_to_states)
        values **= self.weight / weight_sum
        self.tmp_vals = values.transpose()
        return [ts.ckks_vector(context, col) if context else col for col in self.tmp_vals]

    def _calc_col_inner_prods(
            self, enc_cols_parties: list[list[ts.CKKSVector | npt.NDArray[np.float_]]],
            nr_parties: int) -> list[float]:
        col_ips: list[float] = []

        for col_ind, own_col in enumerate(self.tmp_vals):
            res = own_col.tolist()
            for party in enc_cols_parties:
                res *= party[col_ind]
            if isinstance(res, ts.CKKSVector):
                res_dec, *_ = cast(list[float], res.sum().decrypt())
            else:
                res_dec = res.sum()
            col_ips.append(res_dec ** (1 / nr_parties))

        return col_ips

    def get_combined_cpd(
            self, node: str, column_sums: list[float],
            parents_union: list[str], node_to_states: dict[str, list[str]]) -> TabularCPD:
        values = np.array([v / column_sums[i] for i, v in enumerate(self.tmp_vals)]).transpose()
        return TabularCPD(
            variable=node,
            variable_card=len(node_to_states[node]),
            values=values,
            evidence=parents_union,
            evidence_card=[len(node_to_states[p]) for p in parents_union],
            state_names=node_to_states
        )

    def _get_expanded_values(
            self, node: str, node_to_states: dict[str, list[str]]) -> npt.NDArray[np.float_]:
        factor = cast(TabularCPD, self.local_bn.get_cpds(node)).to_factor()
        values: list[list[float]] = []
        parent_to_states = node_to_states.copy()
        del parent_to_states[node]
        parent_states_combinations = list(itertools.product(*list(parent_to_states.values())))

        for node_state in node_to_states[node]:
            col: list[float] = []

            # TODO generalize for when a node or evidence state is not available
            for parent_states_combination in parent_states_combinations:
                get_value_dict = {
                    parent: parent_states_combination[i]
                    for i, parent in enumerate(parent_to_states)
                    if parent in factor.variables
                }
                get_value_dict[node] = node_state
                col.append(cast(float, factor.get_value(**get_value_dict)))
            values.append(col)

        return np.array(values)

    def share_values(
            self, vals: npt.NDArray[np.float_], n_shares: int) -> list[npt.NDArray[np.float_]]:
        shares: list[npt.NDArray[np.float_]] = []
        final_share = vals.copy()

        for _ in range(n_shares - 1):
            share = self.rand_gen.uniform(low=MIN_VAL, high=1.0, size=vals.shape)
            shares.append(share)
            final_share /= share

        shares.append(final_share)

        return shares

    def mark_overlap_solved(self, node: str, ov_nodes: Collection[str],
                            cpd: TabularCPD | None) -> None:
        self.solved_overlaps.add(node)
        self.no_marg_nodes.extend(ov_nodes)

        if cpd:
            self.node_to_cpd[cpd.variable] = cpd
            self.node_to_fact[cpd.variable] = DiscFact.from_cpd(cpd, self.dfc)
            self.node_to_nr_states.update((n, len(s)) for n, s in cpd.state_names.items())
        else:
            del self.node_to_cpd[node]
            del self.node_to_fact[node]

    def update_dfc(self, dfc: DiscFactCfg) -> None:
        for party in cast(Party, self), *self.peers:
            super(Party, party).update_dfc(dfc)
            for fact in party.node_to_fact.values():
                fact.set_cfg(dfc)

    def update_veh(self, veh: VarElimHeur) -> None:
        for party in cast(Party, self), *self.peers:
            super(Party, party).update_veh(veh)


def combine(
        bns: list[BayesianNetwork], weights: list[float], split_ov: bool,
        dfc: DiscFactCfg, veh: VarElimHeur) -> Party:
    if len(bns) != len(weights):
        raise ValueError("Length of bayesian network and weight lists should be equal.")
    parties = [Party(i, bn, weights[i], split_ov, dfc, veh) for i, bn in enumerate(bns)]

    for party in parties:
        party.add_peers(parties)

    next(iter(parties)).combine()

    return min(parties, key=lambda x: x.identifier)
