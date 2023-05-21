from typing import cast

import numpy as np
from model import Model
from sklearn.metrics import f1_score


def sf1_score(true_model: Model, estimated_model: Model) -> float:
    true_adj = true_model.adj_mat(False)
    estimated_adj = estimated_model.adj_mat(False)

    return cast(float, f1_score(np.ravel(true_adj), np.ravel(estimated_adj)))


# https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/master/cdt/metrics.py
def shd_score(true_model: Model, estimated_model: Model, double_for_anticausal=True) -> float:
    true_adj = true_model.adj_mat(True)
    estimated_adj = estimated_model.adj_mat(True)

    diff = np.abs(true_adj - estimated_adj)

    if double_for_anticausal:
        return np.sum(diff)

    diff = diff + diff.transpose()
    diff[diff > 1] = 1

    return float(np.sum(diff) / 2)
