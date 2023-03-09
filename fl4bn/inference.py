from typing import Collection, cast

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork


def elimination_ask(
        bayes_net: BayesianNetwork,
        query: set[str],
        evidence: dict[str, str]) -> DiscreteFactor:
    factors = [
        cast(
            DiscreteFactor,
            cpd.to_factor().reduce(
                [(var, evidence[var]) for var in cpd.variables if var in evidence],
                inplace=False,
                show_warnings=True
            )
        )
        for cpd in cast(list[TabularCPD], bayes_net.get_cpds())
    ]
    node_to_factors: dict[str, set[DiscreteFactor]] = {
        node: set() for node in cast(Collection[str], bayes_net.nodes())
    }

    for factor in factors:
        for variable in cast(list[str], factor.variables):
            node_to_factors[variable].add(factor)

    for var in cast(Collection[str], bayes_net.nodes()):
        if var in evidence or var in query:
            continue

        relevant_factors = node_to_factors[var].copy()

        factors = [factor for factor in factors if factor not in relevant_factors]

        relevant_factors_iter = iter(relevant_factors)
        product_relevant_factors = next(relevant_factors_iter).copy()

        for relevant_factor in relevant_factors_iter:
            product_relevant_factors.product(relevant_factor, inplace=True)
        product_relevant_factors.marginalize([var], inplace=True)

        for node_factors in node_to_factors.values():
            node_factors -= relevant_factors

        for prf_var in cast(list[str], product_relevant_factors.variables):
            node_to_factors[prf_var].add(product_relevant_factors)

        factors.append(product_relevant_factors)

    product_factors = factors[0].copy()
    for factor in factors[1:]:
        product_factors.product(factor, inplace=True)
    product_factors.normalize(inplace=True)

    return product_factors


def disjoint_elimination_ask(
        bayes_net: BayesianNetwork,
        query: set[str],
        evidence: dict[str, str]) -> dict[str, DiscreteFactor]:
    factor = elimination_ask(bayes_net, query, evidence)
    return {
        var: cast(DiscreteFactor, factor.marginalize(query - {var}, inplace=False))
        for var in query
    }


def map_elimination_ask(
        bayes_net: BayesianNetwork,
        query: set[str],
        evidence: dict[str, str]) -> dict[str, str]:
    factor = elimination_ask(bayes_net, query, evidence)
    argmax = np.argmax(factor.values)
    assignment, *_ = cast(list[list[tuple[str, str]]], factor.assignment([argmax]))
    return dict(assignment)


def elimination_ask_alt(
        bayes_net: BayesianNetwork,
        query: set[str],
        evidence: dict[str, str]) -> DiscreteFactor:
    factors: list[DiscreteFactor] = []

    for node in cast(Collection[str], bayes_net.nodes()):
        factor = cast(TabularCPD, bayes_net.get_cpds(node)).to_factor()
        factor.reduce(
            [(var, evidence[var]) for var in factor.variables if var in evidence],
            inplace=True,
            show_warnings=True
        )
        factors.append(factor)

    product_factors = factors[0].copy()
    for factor in factors[1:]:
        product_factors.product(factor, inplace=True)
    product_factors.marginalize(
        [v for v in product_factors.variables if v not in query],
        inplace=True
    )
    product_factors.normalize(inplace=True)

    return product_factors
