import itertools
from collections import defaultdict
from typing import cast

import numpy as np
import numpy.typing as npt
import private_set_intersection.python as psi
import tenseal as ts
from pgmpy.factors.discrete.CPD import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork

REVEAL_INTERSECTION = True
FPR = 0.0
DS = psi.DataStructure.RAW
PMD_TO_MAX_CM_BITS = {1024: 27, 2048: 54, 4096: 109, 8192: 218, 16384: 438, 32768: 881}
HE_DEC_BITS = 40
SMPC_SEED = 1
MIN_VAL = 1e-3


class Client:
    def __init__(self, identifier: int, local_bn: BayesianNetwork) -> None:
        self.identifier = identifier
        self.local_bn = local_bn
        self.cpd_nodes: list[str] = sorted(local_bn.nodes)
        self.node_to_cpd: dict[str, TabularCPD] = {
            cpd.variable: cpd for cpd in cast(list[TabularCPD], local_bn.get_cpds())
        }
        self.clients: list[Client] = []
        self.node_to_neighbors: dict[str, list[Client]] = defaultdict(list)
        self.solved_overlaps: set[str] = set()
        self.rand_gen = np.random.default_rng(seed=SMPC_SEED)
        self.tmp_vals: npt.NDArray[np.float_] = np.array([])

    def add_clients(self, clients: list['Client']) -> None:
        self.clients = sorted(
            [client for client in clients if client.identifier != self.identifier],
            key=lambda x: x.identifier
        )

    def combine(self) -> None:
        self.find_overlaps()
        for client in self.clients:
            client.find_overlaps()

        self.solve_overlaps()
        for client in self.clients:
            client.solve_overlaps()

    def find_overlaps(self) -> None:
        own_nodes_list = list(self.local_bn.nodes)
        for client in self.clients:
            if client.identifier < self.identifier:
                continue

            client_nodes_list = list(client.local_bn.nodes)
            psi_c = psi.client.CreateWithNewKey(REVEAL_INTERSECTION)
            psi_s = psi.server.CreateWithNewKey(REVEAL_INTERSECTION)
            setup = psi.ServerSetup()
            setup.ParseFromString(
                psi_s.CreateSetupMessage(
                    FPR, len(client_nodes_list), own_nodes_list, DS
                ).SerializeToString()
            )
            request = psi.Request()
            request.ParseFromString(
                psi_c.CreateRequest(client_nodes_list).SerializeToString()
            )
            response = psi.Response()
            response.ParseFromString(psi_s.ProcessRequest(request).SerializeToString())
            intersect: list[str] = [
                client_nodes_list[i] for i in psi_c.GetIntersection(setup, response)
            ]
            self.update_overlaps(client, intersect)
            client.update_overlaps(self, intersect)

        self.node_to_neighbors = {
            node: sorted(set(neighbors), key=lambda x: x.identifier)
            for node, neighbors in self.node_to_neighbors.items()
        }

    def update_overlaps(self, client: 'Client', intersect: list[str]) -> None:
        for node in intersect:
            self.node_to_neighbors[node].append(client)

    def solve_overlaps(self) -> None:
        for node in self.node_to_neighbors:
            if node in self.solved_overlaps:
                continue

            neighbors = self.node_to_neighbors[node]
            overlap_clients: list[Client] = [self, *neighbors]
            parents_union = self.get_parents_union(node, overlap_clients)
            nr_parties = len(overlap_clients)
            node_to_states = self.get_node_to_states([node, *parents_union], overlap_clients)
            context = self.gen_context(nr_parties, len(node_to_states[node]))
            self.set_vals_ret_enc(node, nr_parties, node_to_states, context)
            enc_cols_clients = [
                client.set_vals_ret_enc(node, nr_parties, node_to_states, context)
                for client in neighbors
            ]
            column_sums: list[float] = self.calc_col_ips(enc_cols_clients, nr_parties)

            client_unmixed_shares = [
                client.share_values(client.tmp_vals, len(overlap_clients))
                for client in overlap_clients
            ]
            for i, shares in enumerate(map(list, zip(*client_unmixed_shares))):
                overlap_clients[i].tmp_vals = np.array([1.0])
                for share in shares:
                    overlap_clients[i].tmp_vals = overlap_clients[i].tmp_vals * share

            for client in overlap_clients:
                client.set_combined_cpd(node, column_sums, parents_union, node_to_states)
                client.mark_overlap_solved(node)

    def get_parents_union(self, node: str, clients: list['Client']) -> list[str]:
        return sorted(set().union(*[client.local_bn.get_parents(node) for client in clients]))

    def get_node_to_states(
            self, nodes: list[str], clients: list['Client']) -> dict[str, list[str]]:
        node_to_val_dict: dict[str, dict[str, None]] = {node: {} for node in nodes}

        for client in clients:
            for node in nodes:
                if node not in client.local_bn.states:
                    continue
                node_to_val_dict[node].update(dict.fromkeys(client.local_bn.states[node], None))

        return {node: list(values_dict.keys()) for node, values_dict in node_to_val_dict.items()}

    def gen_context(self, nr_parties: int, nr_states: int) -> ts.Context:
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
            self, node: str, nr_parties: int,
            node_to_states: dict[str, list[str]],
            context: ts.Context | None) -> list[ts.CKKSVector | npt.NDArray[np.float_]]:
        values = self.get_expanded_values(node, node_to_states) ** (1 / nr_parties)
        self.tmp_vals = values.transpose()
        return [ts.ckks_vector(context, col) if context else col for col in self.tmp_vals]

    def calc_col_ips(
            self, enc_cols_clients: list[list[ts.CKKSVector | npt.NDArray[np.float_]]],
            nr_parties: int) -> list[float]:
        col_ips: list[float] = []

        for col_ind, own_col in enumerate(self.tmp_vals):
            res = own_col.tolist()
            for client in enc_cols_clients:
                res *= client[col_ind]
            if isinstance(res, ts.CKKSVector):
                res_dec, *_ = cast(list[float], res.sum().decrypt())
            else:
                res_dec = res.sum()
            col_ips.append(res_dec ** (1 / nr_parties))

        return col_ips

    def set_combined_cpd(
            self, node: str, column_sums: list[float],
            parents_union: list[str], node_to_states: dict[str, list[str]]) -> None:
        values = np.array([v / column_sums[i] for i, v in enumerate(self.tmp_vals)]).transpose()
        cpd = TabularCPD(
            variable=node,
            variable_card=len(node_to_states[node]),
            values=values,
            evidence=parents_union,
            evidence_card=[len(node_to_states[p]) for p in parents_union],
            state_names=node_to_states
        )
        self.node_to_cpd[cpd.variable] = cpd

    def get_expanded_values(
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

    def mark_overlap_solved(self, node: str) -> None:
        self.solved_overlaps.add(node)

    def elimination_ask(
            self, query: list[str], evidence: dict[str, str], propagate=True) -> DiscreteFactor:
        no_marg_vars: set[str] = set()
        if propagate:
            factors = [client.elimination_ask(query, evidence, False) for client in self.clients]
            factors.append(self.elimination_ask(query, evidence, False))
            nodes: list[str] = sorted(set(var for factor in factors for var in factor.variables))
        else:
            factors = [
                cast(
                    DiscreteFactor,
                    cpd.to_factor().reduce(
                        [(var, evidence[var]) for var in cpd.variables if var in evidence],
                        inplace=False,
                        show_warnings=False
                    )
                )
                for cpd in self.node_to_cpd.values()
            ]
            nodes = self.cpd_nodes
            no_marg_vars = set(
                v for n in self.node_to_neighbors for v in self.node_to_cpd[n].variables
            )

        node_to_factors: dict[str, list[DiscreteFactor]] = defaultdict(list)

        for factor in factors:
            for variable in cast(list[str], factor.variables):
                node_to_factors[variable].append(factor)

        for var in nodes:
            if var in evidence or var in query:
                continue

            relevant_factors = node_to_factors[var].copy()

            factors = [factor for factor in factors if factor not in relevant_factors]

            product_relevant_factors = DiscreteFactor([], [], [1])

            for relevant_factor in relevant_factors:
                product_relevant_factors.product(relevant_factor, inplace=True)

            if var not in no_marg_vars:
                product_relevant_factors.marginalize([var], inplace=True)

            for node_factors in node_to_factors.values():
                for rel_fact in relevant_factors:
                    if rel_fact in node_factors:
                        node_factors.remove(rel_fact)

            for prf_var in cast(list[str], product_relevant_factors.variables):
                node_to_factors[prf_var].append(product_relevant_factors)

            factors.append(product_relevant_factors)

        product_factors = DiscreteFactor([], [], [1])

        for factor in factors:
            product_factors.product(factor, inplace=True)

        product_factors.normalize(inplace=True)

        return product_factors

    def disjoint_elimination_ask(
            self, query: list[str], evidence: dict[str, str]) -> dict[str, DiscreteFactor]:
        factor = self.elimination_ask(query, evidence)

        return {
            var: cast(
                DiscreteFactor, factor.marginalize([q for q in query if q != var], inplace=False))
            for var in query
        }

    def map_elimination_ask(self, query: list[str], evidence: dict[str, str]) -> dict[str, str]:
        factor = self.elimination_ask(query, evidence)
        argmax = np.argmax(factor.values)
        assignment, *_ = cast(list[list[tuple[str, str]]], factor.assignment([argmax]))

        return dict(assignment)


def combine(bns: list[BayesianNetwork]) -> Client:
    clients = [Client(i, bn) for i, bn in enumerate(bns)]

    for client in clients:
        client.add_clients(clients)

    next(iter(clients)).combine()

    return min(clients, key=lambda x: x.identifier)
