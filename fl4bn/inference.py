import itertools
from collections import defaultdict
from typing import Collection, cast

import numpy as np
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.models import BayesianNetwork


def elimination_ask(
        source: BayesianNetwork | list[BayesianNetwork],
        query: list[str],
        evidence: dict[str, str],
        marg=True) -> DiscreteFactor:
    if isinstance(source, list):
        factors = [elimination_ask(bn, query, evidence, False) for bn in source]
        # for factor in factors:
        #     print(factor)
        #     print("/")
        # print()
        no_merge_vars: set[str] = set()
    else:
        factors = [
            cast(
                DiscreteFactor,
                cpd.to_factor().reduce(
                    [(var, evidence[var]) for var in cpd.variables if var in evidence],
                    inplace=False,
                    show_warnings=True
                )
            )
            for cpd in cast(list[TabularCPD], source.get_cpds())
        ]
        if marg:
            no_merge_vars: set[str] = set()
        else:
            vars_with_cpd: set[str] = {
                cpd.variable for cpd in cast(list[TabularCPD], source.get_cpds())
            }
            no_merge_vars: set[str] = set()
            for out, inc in cast(Collection[tuple[str, str]], source.edges()):
                set_out_inc = set([out, inc])
                if len(set_out_inc & vars_with_cpd) == 1:
                    if source.in_degree(out) and source.out_degree(out):
                        no_merge_vars.add(out)
                    if source.in_degree(inc) and source.out_degree(inc):
                        no_merge_vars.add(inc)

    node_to_factors: dict[str, list[DiscreteFactor]] = defaultdict(list)

    for factor in factors:
        for variable in cast(list[str], factor.variables):
            node_to_factors[variable].append(factor)

    for var in node_to_factors:
        if var in evidence or var in query:
            continue

        relevant_factors = node_to_factors[var].copy()

        factors = [factor for factor in factors if factor not in relevant_factors]

        product_relevant_factors = DiscreteFactor([], [], [1])

        for relevant_factor in relevant_factors:
            product_relevant_factors.product(relevant_factor, inplace=True)

        if var not in no_merge_vars:
            product_relevant_factors.marginalize([var], inplace=True)

        for node_factors in node_to_factors.values():
            for relevant_fact in relevant_factors:
                if relevant_fact in node_factors:
                    node_factors.remove(relevant_fact)

        for prf_var in cast(list[str], product_relevant_factors.variables):
            node_to_factors[prf_var].append(product_relevant_factors)

        factors.append(product_relevant_factors)

    product_factors = DiscreteFactor([], [], [1])

    for factor in factors:
        product_factors.product(factor, inplace=True)

    product_factors.normalize(inplace=True)

    return product_factors


def disjoint_elimination_ask(
        source: BayesianNetwork | list[BayesianNetwork],
        query: list[str],
        evidence: dict[str, str]) -> dict[str, DiscreteFactor]:
    factor = elimination_ask(source, query, evidence)
    return {
        var: cast(
            DiscreteFactor, factor.marginalize([q for q in query if q != var], inplace=False))
        for var in query
    }


def map_elimination_ask(
        source: BayesianNetwork | list[BayesianNetwork],
        query: list[str],
        evidence: dict[str, str]) -> dict[str, str]:
    factor = elimination_ask(source, query, evidence)
    argmax = np.argmax(factor.values)
    assignment, *_ = cast(list[list[tuple[str, str]]], factor.assignment([argmax]))
    return dict(assignment)
