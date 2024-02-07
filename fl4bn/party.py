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
            cpd.variable: cpd for cpd in cast(list[TabularCPD], local_bn.get_cpds())}
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
        # variable to keep track of communication size
        self.last_nr_comm_vals = 0

        # All parties run a local round of inference
        for party in cast(Party, self), *self.peers:
            # Inference factors are those present in the party after propagating evidence
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
            # Do not eliminate targets, overlaps & their parents, or evidence
            discard: set[str] = set().union(targets, party.no_marg_nodes, evidence)
            # Set the variables to be eliminated during inference
            nodes = set(n for n in party.node_to_cpd if n not in discard)

            # res_fact = self.var_elim(party_facts, nodes, party.node_to_nr_states)
            # facts.append(res_fact)
            # self.last_nr_comm_vals += res_fact.values.size

            res_facts = self.var_elim(party_facts, nodes, party.node_to_nr_states, True)
            facts.extend(res_facts)
            self.last_nr_comm_vals += sum(f.values.size for f in res_facts)

        nodes: set[str] = set()
        node_to_nr_states: dict[str, int] = {}

        # Gather variables present in intermediary factors and their state counts
        for fact in facts:
            nodes.update(fact.variables)
            node_to_nr_states.update((n, len(s)) for n, s in fact.state_names.items())

        nodes.difference_update(targets)

        # Querying party runs final variable elimination
        return self.var_elim(facts, nodes, node_to_nr_states)

    def as_dig(self) -> nx.DiGraph:
        # Returns the directed graph that would result from putting all parties together
        dig = nx.DiGraph()

        for party in [cast(Party, self), *self.peers]:
            dig.add_edges_from(
                [(p, n) for n, f in party.node_to_cpd.items() for p in f.get_evidence()])

        return dig

    def add_peers(self, parties: list['Party']) -> None:
        # Ensures multiple peers with the same identifier can no exists
        self.peers.extend(p for p in parties if p.identifier != self.identifier)
        # Sorts peers based on identifier, only for easier inspection
        self.peers = sorted(set(self.peers), key=attrgetter("identifier"))

    def combine(self) -> None:
        for party in cast(Party, self), *self.peers:
            party.find_overlaps()

        for party in cast(Party, self), *self.peers:
            party.solve_overlaps()

    def find_overlaps(self) -> None:
        own_nodes_list = list(self.local_bn.nodes)
        for party in self.peers:
            # Avoids running intersections between a pair of parties twice
            if party.identifier < self.identifier:
                continue

            party_nodes_list = list(party.local_bn.nodes)
            # Set up private set intersection procedure
            # Any of the two parties acts as the "client", while the other is the "server"
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
            # Run intersection and get results
            intersect: list[str] = [
                party_nodes_list[i] for i in psi_c.GetIntersection(setup, response)
            ]
            # The two parties update their state based on the intersection result
            self.update_overlaps(party, intersect)
            party.update_overlaps(self, intersect)

        # After finding all overlaps for a party, they are sorted, only for easier inspection
        self.node_to_neighbors = {
            node: sorted(set(neighbors), key=lambda x: x.identifier)
            for node, neighbors in self.node_to_neighbors.items()
        }

    def update_overlaps(self, party: 'Party', intersect: list[str]) -> None:
        for node in intersect:
            self.node_to_neighbors[node].append(party)

    def solve_overlaps(self) -> None:
        for node in self.node_to_neighbors:
            # Skip already solved overlaps
            if node in self.solved_overlaps:
                continue

            neighbors = self.node_to_neighbors[node]
            overlap_parties: list[Party] = [self, *neighbors]
            parents_union = self._get_parents_union(node, overlap_parties)
            nr_parties = len(overlap_parties)
            weight_sum = sum(p.weight for p in overlap_parties)
            # Get the states for all nodes involved in overlap from participating parties
            node_to_states = self._get_node_to_states([node, *parents_union], overlap_parties)
            # Set up homomorphic encryption
            # context = None
            context = self._gen_context(nr_parties, len(node_to_states[node]))
            # Party that initiated overlap solving calculates its updated CPD values
            self.set_vals_ret_enc(node, weight_sum, node_to_states, context)
            # Other overlap parties calcuate values for their updated CPDs
            # The encrypted transposed results are sent to the initiator
            enc_cols_parties = [
                party.set_vals_ret_enc(node, weight_sum, node_to_states, context)
                for party in neighbors
            ]
            # Initiator calculates normalization values for the columns
            column_sums: list[float] = self._calc_col_inner_prods(enc_cols_parties, nr_parties)

            # Create multiplication secret shares
            party_unmixed_shares = [
                party.share_values(party.tmp_vals, nr_parties)
                for party in overlap_parties
            ]

            # Distribute secret shares across parties
            for i, shares in enumerate(map(list, zip(*party_unmixed_shares))):
                overlap_parties[i].tmp_vals = np.array([1.0])
                for share in shares:
                    overlap_parties[i].tmp_vals = overlap_parties[i].tmp_vals * share

            # If runing the more secure protocol, keeping overlap CPDs distributed
            if self.split_ov:
                for party in overlap_parties:
                    # Parties form their local CPD object
                    cpd = party.get_combined_cpd(node, column_sums, parents_union, node_to_states)
                    # Parties update their state after solving overlap
                    party.mark_overlap_solved(node, node_to_states, cpd)
            # If running the faster protocol, combining overlap CPDs in one party
            else:
                # Party holding CPD creates the local object for it
                cpd = self.get_combined_cpd(node, column_sums, parents_union, node_to_states)

                for party in neighbors:
                    # Party holding CPD performs product with CPD values from other parties
                    cpd.values *= party.get_combined_cpd(
                        node, column_sums, parents_union, node_to_states).values
                    # Other parties update their state, but do not hold on to any CPD
                    party.mark_overlap_solved(node, node_to_states, None)
                # Party holding CPD undates its state, accounting for the held CPD
                self.mark_overlap_solved(node, node_to_states, cpd)

        # Sorting the nodes that should not be marginalized is only done for easier inspection
        self.no_marg_nodes = sorted(set(self.no_marg_nodes))

    def _get_parents_union(self, node: str, parties: list['Party']) -> list[str]:
        # Get the union of all parents across parties for the node
        # Sorting is not strictly required
        # However, other implementation parts assume order is the same for all parties
        return sorted(set().union(*[party.local_bn.get_parents(node) for party in parties]))

    def _get_node_to_states(
            self, nodes: list[str], parties: list['Party']) -> dict[str, list[str]]:
        node_to_val_dict: dict[str, dict[str, None]] = {node: {} for node in nodes}

        # Build a list of states for all nodes in the overlap
        for party in parties:
            for node in nodes:
                if node not in party.local_bn.states:
                    continue
                node_to_val_dict[node].update(dict.fromkeys(party.local_bn.states[node], None))

        return {node: list(values_dict) for node, values_dict in node_to_val_dict.items()}

    def _gen_context(self, nr_parties: int, nr_states: int) -> ts.Context:
        # Select a list of bit sizes for primes used during the computation knowing that
        # The outer (first and last) elements must be equal
        # The inner (other) elements must be equal
        # The number of inner elements is that of the max consecutive multplications
        # Result precision (bits) before decimal point is outer - inner
        # Result precision after decimal point is 2 * inner - outer
        # For calculating normalization values the following also holds
        # The number of parties gives the number of consecutive multiplications
        # All input CPD entries are <1
        # Consequently, the maximum normalization value is the number of states
        aux = nr_states.bit_length()
        inner_bits = HE_DEC_BITS + aux
        outer_bits = inner_bits + aux
        cmb_sizes = [outer_bits, *([inner_bits] * nr_parties), outer_bits]
        # The sum of the list elements gives the coefficient modulus bit count
        cm_bits = sum(cmb_sizes)
        pm_degree = 0
        # Find the smallest fitting polynomial modulus
        # Specifically its max coefficient modulus bit count should be greater than the current one
        for pmd, max_cm_bits in PMD_TO_MAX_CM_BITS.items():
            if cm_bits < max_cm_bits:
                pm_degree = pmd
                break
        context = ts.context(ts.SCHEME_TYPE.CKKS,
                             poly_modulus_degree=pm_degree, coeff_mod_bit_sizes=cmb_sizes)
        context.generate_galois_keys()
        # Multiplication scale should be 2 raised to the power of the inner bit count
        context.global_scale = 2**inner_bits
        return context

    def set_vals_ret_enc(
            self, node: str, weight_sum: float,
            node_to_states: dict[str, list[str]],
            context: ts.Context | None) -> list[ts.CKKSVector | npt.NDArray[np.float_]]:
        # Expand CPD shape to accomodate additional parents by repeating values
        values = self._get_expanded_values(node, node_to_states)
        # Partially apply geometric mean
        values **= self.weight / weight_sum
        # Transpose into columns and store in temporary variable
        self.tmp_vals = values.transpose()
        # Encrypt before returning if homomorphic encryption is set up
        return [ts.ckks_vector(context, col) if context else col for col in self.tmp_vals]

    def _calc_col_inner_prods(
            self, enc_cols_parties: list[list[ts.CKKSVector | npt.NDArray[np.float_]]],
            nr_parties: int) -> list[float]:
        col_ips: list[float] = []

        # Iterate through columns
        for col_ind, own_col in enumerate(self.tmp_vals):
            res = own_col.tolist()
            # Multiply the corresponding party columns element-wise
            for party in enc_cols_parties:
                res *= party[col_ind]
            # Decrypt after summing vector entries if dealing with encrypted data
            if isinstance(res, ts.CKKSVector):
                res_dec, *_ = cast(list[float], res.sum().decrypt())
            # If data is unencrypted, merely sum vector entries
            else:
                res_dec = res.sum()
            # Raise normalization value to the root of the number of parties
            # Allows parties to all apply it directly
            col_ips.append(res_dec ** (1 / nr_parties))

        return col_ips

    def get_combined_cpd(
            self, node: str, column_sums: list[float],
            parents_union: list[str], node_to_states: dict[str, list[str]]) -> TabularCPD:
        # Apply normalization values to columns and transpose back from columns
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
        # Get all possible state assignments for parents and child in expanded CPD
        parent_states_combinations = list(itertools.product(*list(parent_to_states.values())))

        # For each CPD row (i.e., child state)
        for node_state in node_to_states[node]:
            col: list[float] = []

            # For each CPD column (i.e., parent asisgnment)
            for parent_states_combination in parent_states_combinations:
                # Get corresponding entry from initial CPD
                # Assumes modeled variables have the same states in other parties
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

        # For all but the last required share
        for _ in range(n_shares - 1):
            # Sample its entries
            share = self.rand_gen.uniform(low=MIN_VAL, high=1.0, size=vals.shape)
            shares.append(share)
            # Update the final share
            final_share /= share

        shares.append(final_share)

        return shares

    def mark_overlap_solved(self, node: str, ov_nodes: Collection[str],
                            cpd: TabularCPD | None) -> None:
        self.solved_overlaps.add(node)
        self.no_marg_nodes.extend(ov_nodes)

        # If storing a CPD, assign its new information to the corresponding node
        if cpd:
            self.node_to_cpd[cpd.variable] = cpd
            self.node_to_fact[cpd.variable] = DiscFact.from_cpd(cpd, self.dfc)
            self.node_to_nr_states.update((n, len(s)) for n, s in cpd.state_names.items())
        # If not storing a CPD, delete its old information for the corresponding node
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

    # Take the first party in the list and initiate the inference preparation process (CABN)
    # Any other party could be chosen to initiate the procedure
    next(iter(parties)).combine()

    # Return the party with the lowest identifier
    # Any other party could be returned
    return min(parties, key=lambda x: x.identifier)
