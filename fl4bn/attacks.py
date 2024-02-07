import itertools
import warnings
from collections import Counter
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from disc_fact import DiscFactCfg
from experiment import _split_vars, _train_model, sample
from party import Party, combine
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.utils import get_example_model
from var_elim_heurs import MinWeightVEH


# Works around pgmpy issues with products over CPDs.
def prod(phi0: TabularCPD, phi1: TabularCPD) -> TabularCPD:
    phi = phi0.copy()
    if isinstance(phi1, (int, float)):
        phi.values *= phi1
    else:
        # Compute the new values
        set_phi_vars = set(phi.variables)
        new_variables = phi.variables.copy()
        new_variables.extend(v for v in phi1.variables if v not in set_phi_vars)
        var_to_int = {var: index for index, var in enumerate(new_variables)}
        phi.values = np.einsum(
            phi.values,
            [var_to_int[var] for var in phi.variables],
            phi1.values,
            [var_to_int[var] for var in phi1.variables],
            range(len(new_variables)),
        )

        # Compute the new cardinality array
        phi_card = {var: card for var, card in zip(phi.variables, phi.cardinality)}
        phi1_card = {
            var: card for var, card in zip(phi1.variables, phi1.cardinality)
        }
        phi_card.update(phi1_card)
        phi.cardinality = np.array([phi_card[var] for var in new_variables])

        # Set the new variables and state names
        phi.variables = new_variables
        phi.add_state_names(phi1)

    return phi


def prep_models(
        ref_bn: BayesianNetwork,
        overlap_ratio: float,
        split_ov: bool,
        samples_factor: int,
        connected: bool,
        r_seed: int | None) -> tuple[Party, Party, list[str]]:
    nr_clients = 2
    nr_samples_per_client = round(samples_factor * len(ref_bn.nodes()) // nr_clients)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        all_samples = sample(ref_bn, nr_samples_per_client * nr_clients, r_seed)
        clients_train_samples: list[pd.DataFrame] = []
        running_sum = 0
        for _ in range(nr_clients):
            clients_train_samples.append(
                all_samples[running_sum:running_sum + nr_samples_per_client])
            running_sum += nr_samples_per_client
        (_, max_in_deg), *_ = Counter(e_inc for (_, e_inc, *_) in ref_bn.edges()).most_common(1)
        clients_train_vars, ov_vars = _split_vars(
            ref_bn, nr_clients, overlap_ratio, connected, seed=r_seed)
        trained_models = [
            _train_model(clients_train_samples[i][train_vars], max_in_deg, ref_bn.states)
            for i, train_vars in enumerate(clients_train_vars)
        ]
        eq_weights = [1.0] * len(trained_models)
        dfc = DiscFactCfg(False, np.float_)
        veh = MinWeightVEH()
        party = combine(trained_models, eq_weights, split_ov, dfc, veh)
        peer, *_ = party.peers
        return party, peer, ov_vars


def reconstruct_cpd_from_comb(
        own_in_cpd: TabularCPD, comb_cpd: TabularCPD, tgt_cpd_evid: set[str]) -> TabularCPD:
    loc_cpd = own_in_cpd.copy()
    loc_cpd.values **= 0.5
    reconstr_cpd = cast(TabularCPD, comb_cpd / loc_cpd)
    # print(reconstr_cpd)
    reconstr_cpd.marginalize(
        [v for v in reconstr_cpd.get_evidence() if v not in tgt_cpd_evid],
        inplace=True)
    reconstr_cpd.values **= 2
    reconstr_cpd.normalize(inplace=True)

    return reconstr_cpd


def main() -> None:
    model = get_example_model("asia")
    overlap_ratio = 0.1

    # Augment Attack
    p_1, p_2, ov_vars = prep_models(model, overlap_ratio, False, 500, True, 42)

    for tgt_var in ov_vars:
        tgt_cpd = cast(TabularCPD, p_2.local_bn.get_cpds(tgt_var))

        comb_cpd = p_1.node_to_cpd[tgt_var]

        reconstr_cpd = reconstruct_cpd_from_comb(
            cast(TabularCPD, p_1.local_bn.get_cpds(tgt_var)),
            comb_cpd,
            set(tgt_cpd.get_evidence()))

        if reconstr_cpd != tgt_cpd:
            raise RuntimeError(f"Reconstruction of {tgt_var} failed!")

    # Inference Attack
    p_1, p_2, ov_vars = prep_models(model, overlap_ratio, True, 500, True, 42)

    for tgt_var in ov_vars:
        tgt_cpd = cast(TabularCPD, p_2.local_bn.get_cpds(tgt_var))

        own_aug_cpd = p_1.node_to_cpd[tgt_var]
        parent_to_states = own_aug_cpd.state_names.copy()
        del parent_to_states[tgt_var]
        parent_states_combinations = list(itertools.product(*list(parent_to_states.values())))
        values: list[npt.NDArray[np.float_]] = []
        for parent_states_combination in parent_states_combinations:
            evid = {
                parent: parent_states_combination[i]
                for i, parent in enumerate(parent_to_states)
            }
            values.append(p_1.query([tgt_var], evid).values)
        var_to_card = own_aug_cpd.get_cardinality(own_aug_cpd.variables)
        parents = [v for v in own_aug_cpd.variables if v != tgt_var]
        comb_cpd = TabularCPD(
            tgt_var,
            var_to_card[tgt_var],
            np.column_stack(values),
            parents,
            [var_to_card[p] for p in parents],
            own_aug_cpd.state_names)

        reconstr_cpd = reconstruct_cpd_from_comb(
            cast(TabularCPD, p_1.local_bn.get_cpds(tgt_var)),
            comb_cpd,
            set(tgt_cpd.get_evidence()))

        if reconstr_cpd != tgt_cpd:
            raise RuntimeError(f"Reconstruction of {tgt_var} failed!")


if __name__ == "__main__":
    main()
