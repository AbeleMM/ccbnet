import itertools
from collections import defaultdict
from enum import Enum, auto
from typing import cast

import numpy as np
import numpy.typing as npt
from disc_fact import DiscFactCfg
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from single_net import SingleNet


class CombineMethod(Enum):
    MULTI = "Multi"
    SINGLE = "Single"
    UNION = "Union"


class CombineOp(Enum):
    SUPERPOS = auto()
    ARITH_MEAN = auto()
    GEO_MEAN = auto()


def combine_bns(
        bns: list[BayesianNetwork], weights: list[float],
        method: CombineMethod, allow_loops: bool, combine_op: CombineOp,
        dfc: DiscFactCfg) -> SingleNet:
    # Assumes nodes identified by strings & same nodes in different networks have same values
    # TODO potentially use class

    if len(bns) != len(weights):
        raise ValueError("Length of bayesian network and weight lists should be equal.")

    bn_to_conf = dict(zip(bns, weights))

    if method is CombineMethod.MULTI:
        return _combine_bns_weighted_multi(bn_to_conf, allow_loops, combine_op, dfc)

    node_to_bns: defaultdict[str, list[BayesianNetwork]] = defaultdict(list)
    node_to_vals: dict[str, list[int | str]] = {}
    bn_combined = SingleNet(allow_loops, dfc)

    for bayes_net in bn_to_conf:
        node_to_vals.update(bayes_net.states)
        for node in cast(list[str], bayes_net.nodes()):
            node_to_bns[node].append(bayes_net)

    for node, node_bns in node_to_bns.items():
        bn_combined.add_node(node)
        # Begin Algorithm 1
        if len(node_bns) == 1:
            _preserve_from_bn(bn_combined, node, node_bns[0])
        else:
            intersect_node_bns: set[str] = set.intersection(*[
                cast(set[str], set(bn.nodes())) for bn in node_bns
            ])
            node_bn_to_parent_set: dict[BayesianNetwork, set[str]] = {
                bn: set(bn.get_parents(node)) for bn in node_bns
            }
            bns_without_intersect_node_parents: list[BayesianNetwork] = [
                bn
                for bn, parent_set in node_bn_to_parent_set.items()
                if parent_set - intersect_node_bns
            ]
            # Interior Node
            if (
                len(bns_without_intersect_node_parents) < len(node_bn_to_parent_set) and
                method != CombineMethod.UNION
            ):
                # Begin Algorithm 2
                _combine_int_node(
                    bn_combined,
                    node,
                    bns_without_intersect_node_parents,
                    node_bn_to_parent_set,
                    bn_to_conf
                )
                # End Algorithm 2
            # Exterior Node
            else:
                # Begin Algorithm 3
                # TODO split into separate method
                parents_union: list[str] = []
                for parent in sorted(set.union(*node_bn_to_parent_set.values())):
                    try:
                        bn_combined.add_edge(parent, node)
                        parents_union.append(parent)
                    # TODO improve handling of cycles
                    except ValueError:
                        pass

                node_cpd = TabularCPD(
                    variable=node,
                    variable_card=len(node_to_vals[node]),
                    values=_get_ext_node_values(
                        node, parents_union, node_to_vals,
                        {bn: bn_to_conf[bn] for bn in node_bns}, combine_op),
                    evidence=parents_union,
                    evidence_card=[len(node_to_vals[p]) for p in parents_union],
                    state_names={node_: node_to_vals[node_] for node_ in [node, *parents_union]}
                )

                node_cpd.normalize(inplace=True)
                bn_combined.add_cpds(node_cpd)
                # End Algorithm 3
        # End Algorithm 1

    return bn_combined


def _combine_int_node(
        bn_combined: BayesianNetwork,
        node: str,
        bns_without_intersect_node_parents: list[BayesianNetwork],
        node_bn_to_parent_set: dict[BayesianNetwork, set[str]],
        bn_to_conf: dict[BayesianNetwork, float]) -> None:
    if len(bns_without_intersect_node_parents) == 0:
        bn_perserve = max(
            node_bn_to_parent_set.keys(),
            key=lambda bn: bn_to_conf[bn]
        )
        _preserve_from_bn(bn_combined, node, bn_perserve)
    elif len(bns_without_intersect_node_parents) == 1:
        _preserve_from_bn(bn_combined, node, bns_without_intersect_node_parents[0])
    else:
        bn_perserve = max(
            bns_without_intersect_node_parents,
            key=lambda bn: bn_to_conf[bn]
        )
        _preserve_from_bn(bn_combined, node, bn_perserve)


def _combine_bns_weighted_multi(
        bn_to_conf: dict[BayesianNetwork, float], allow_loops: bool, combine_op: CombineOp,
        dfc: DiscFactCfg) -> SingleNet:
    iter_bn_to_conf = iter(bn_to_conf.items())
    bn_combined, conf_combined = next(iter_bn_to_conf)
    bn_combined = SingleNet.from_bn(bn_combined, allow_loops, dfc)
    for bayes_net, confidence in iter_bn_to_conf:
        bn_combined = combine_bns(
            [bn_combined, bayes_net],
            [conf_combined, confidence],
            CombineMethod.SINGLE,
            allow_loops,
            combine_op,
            dfc
        )
        conf_combined = (conf_combined + confidence) / 2
    return bn_combined


def _preserve_from_bn(tgt_bn: BayesianNetwork, node: str, src_bn: BayesianNetwork) -> None:
    cpd = cast(TabularCPD, src_bn.get_cpds(node)).copy()
    for parent in cast(list[str], src_bn.get_parents(node)):
        try:
            tgt_bn.add_edge(parent, node)
        # TODO improve handling of cycles
        except ValueError:
            cpd.marginalize([parent], inplace=True)
    tgt_bn.add_cpds(cpd)


def _get_superposition(l_val: float, r_val: float) -> float:
    return l_val + r_val - l_val * r_val


def _get_ext_node_values(
        node: str, parents_union: list[str],
        node_to_vals: dict[str, list[int | str]], node_to_conf: dict[BayesianNetwork, float],
        combine_op: CombineOp) -> npt.NDArray[np.float_]:
    nr_node_vals: int = len(node_to_vals[node])
    transp_table: list[list[float]] = []

    for parent_inst in itertools.product(*[node_to_vals[parent] for parent in parents_union]):
        match combine_op:
            case CombineOp.SUPERPOS | CombineOp.ARITH_MEAN:
                tt_row: list[float] = [0] * nr_node_vals
            case CombineOp.GEO_MEAN:
                tt_row: list[float] = [1] * nr_node_vals
        conf_sum = sum(node_to_conf.values())
        for bayes_net, conf in node_to_conf.items():
            cpd: TabularCPD = cast(TabularCPD, bayes_net.get_cpds(node)).copy()
            cpd.reduce(
                [
                    (parent, parent_inst[i])
                    for i, parent in enumerate(parents_union)
                    if parent in cpd.get_evidence()
                ],
                inplace=True
            )
            cpd.marginalize([v for v in cpd.variables if v != node], inplace=True)
            # TODO account for confidence
            match combine_op:
                case CombineOp.SUPERPOS:
                    tt_row = [_get_superposition(v, cpd.values[i]) for i, v in enumerate(tt_row)]
                case CombineOp.ARITH_MEAN:
                    tt_row = [v + conf * cpd.values[i] / conf_sum for i, v in enumerate(tt_row)]
                case CombineOp.GEO_MEAN:
                    tt_row = [
                        v * ((cpd.values[i] ** conf) ** (1 / conf_sum))
                        for i, v in enumerate(tt_row)]
        transp_table.append(tt_row)

    return np.array(transp_table).T
