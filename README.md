# CCBNet

This repository contains supporting experimental code for the study of [**C**onfidentiality-Preserving **C**ollaborative **B**ayesian **N**etworks](http://resolver.tudelft.nl/uuid:192a90ed-f6fc-4d5b-b3bc-7cf9b67e6754), including an implementation of the main concepts behind the _CCBNet_ framework proposed within it. The project is written in Python and its setup requires running the `poetry install` command in an environment with [Poetry](https://python-poetry.org/). All interaction between different parties is simulated sequentially.

## Project Structure

All source files are in the `fl4bn` directory:

* `run.py` - Entry-point for running experiments
* `writer.py` - Write experiment results to terminal/file
* `experiment.py` - Set up experiment(s) splitting a reference network over a number of parties based on additional given parameters
* `model.py` - Abstract base class for all implemented methods allowing analysis on a Bayesian network
* `single_net.py` - Model subclass that wraps classic Bayesian networks
* `uniform.py` - Model subclass that outputs uniform probabilities for analysis requests as a control
* `avg_outs.py` - Model subclass that runs analysis on parties individually, propagating only modeled evidence and taking the final result for each target as the average across all parties modelling it; represents _DOM_ from the study text
* `party.py` - Model subclass corresponding to _CCBNet_; setting `split_ov` to `False` yields the degenerate _CCBNetJ_ variant from the study text
* `combine.py` - Class for generating a single combined network from party models; setting `method` to `CombineMethod.MULTI` and `combine_op` to `CombineOp.SUPERPOS` yields _CC_ from the study, while `CombineMethod.UNION` alongside `CombineOp.GEO_MEAN` yields _CU_
* `struct_scores.py` - Utility class for calculating similarity between two models
* `disc_fact.py` - Subclass for discrete factors with support for GPU-backed, custom-precision arrays
* `var_elim_heurs.py` - Contains different heuristics for greedily determining variable elimination ordering during inference
* `attacks.py` - Demonstrates performing specific attacks on _CCBNet(J)_

## CCBNet Core Implementation (`fl4bn/party.py`)

### Overview

Alongside the class representing `Party` instances running _CCBNet_, the script contains the `combine` helper method, which creates the parties, makes them aware of each other by calling `add_peers` on each, and prepares them for inference by calling the homonymous `combine` on one of the instances, before returning it.

To allow performing all experiments, the returned instance is assumed to be able to run inference by calling `query` involving all features modeled in total by the parties.

The instance methods `combine` & `query` prepare parties for and perform inference corresponding to the _CABN_ and _SAVE_ protocols from the study text.

The initialization of `Party` instances contains the following state fields:

* `identifier` - A number uniquely identifying each party instance
* `local_bn` - A copy of the input Bayesian network from which the instance is created, not modified otherwise
* `weight` - A value between `0` and `1` (default) giving confidence in the party network
* `split_ov` - A flag giving _CCBNet_ when `True` and _CCBNetJ_ when `False`
* `node_to_cpd` - Gives the corresponding updated CPD for each modeled graph node (i.e., feature)
* `node_to_fact` - Gives the factor corresponding to the updated CPD for each modeled graph node
* `node_to_nr_states` - Gives the number of states for each node modeled or known as an overlap parent
* `peers` - A list containing references to all other parties of which the present one is aware
* `node_to_neighbors` - Gives, for each overlapping node, the corresponding parties that also model it
* `solved_overlaps` - Contains all overlapping nodes which have an updated representation, coherent between all parties containing it
* `rand_gen` - A random generator with the seed set as the party identifier, for reproducibility
* `no_marg_nodes` - A list of nodes which should not be marginalized during local inference, namely overlap nodes and their parents
* `tmp_vals` - A temporary array used when updating CPD values of overlap nodes

### CABN (`Party.combine`)

The following constants dictate parts of parties' behaviour when preparing them for inference:

* `REVEAl_INTERSECTION` - Specifies whether the contents should be revealed alongside the length when privately intersecting pairs of party feature sets; it should always be `True`
* `FPR` - Specifies the private set intersection false positive rate; the default value is `0`
* `DS` - Specifies the [private set intersection data structure](https://github.com/OpenMined/PSI/blob/master/private_set_intersection/python/__init__.py); the default value is `RAW`, while `GCS` and `BloomFilter` are the other possible ones
* `PMD_TO_MAX_CM_BITS` - Specifies the homomorphic encryption max coefficient modulus bit count that still ensures correctness for each given polynomial modulus degree; it should always mirror the values specified in the [corresponding documentation](https://github.com/microsoft/APSI#encryption-parameters)
* `HE_DEC_BITS` - Specifies roughly the precision after the decimal point in homomorphic encryption and setting an overly large value slows down encrypted computation, while an overly small one can lead to improper results; the default value is `40`
* `MIN_VAL` - Specifies the mnimum value for a pseudorandomly generated secret share, and a value `>0` should always be used to ensure a share accidentally having the value `0`, which would prevent reconstructing the original secret; the default value is `0.1`

In the preparation process, parties first finds their overlaps via the `find_overlaps` method. For all others with an identifier greater than its own, each party initiates the procedure for finding common features privately. The intersection is disclosed to both parties, which update their `node_to_neighbors` field accordingly, also with the help of the `update_overlaps` method.

After all parties know their overlaps and who they share them with, each solves its own via the `solve_overlaps` method, which iterates over overlaps not present in the `solved_overlaps` field. Within each iteration, the union of parents for the overlap feature are stored in `parents_union` with the help of the `_get_parents_union` method. All states of the overlap feature and its `parents_union` are collected in the `node_to_states` variable via the `_get_node_to_states` method. The `_gen_context` method prepares the homomorphic encryption `context`. Each overlap party then has method `set_vals_ret_enc` store in field `tmp_vals` the columns for the CPD expanded to incorporate all features/states from `node_to_states`, with entries raised to the power used within the (weighted) geometric mean, and returns an encrypted representation. Setting the `context` to `None` allows skipping the encryption of returned value for debugging purposes. Following that, the method `_calc_col_inner_prods` calculates and decrypts the normalization values of columns into the `column_sums` variable, by multiplying corresponding columns from different parties element-wise and summing up the result's entries. Afterward, parties create the required number of secret shares from their `tmp_vals` field via the `share_values` method and distribute them, updating `tmp_vals` to be the element-wise product of their newly assigned shares. Parties continue by creating their overlap node CPD instance with values corresponding to the columns of `tmp_vals` normalized by `column_sums`, parents corresponding to `parents_union` and states corresponding to `node_to_states`. If the `split_ov` field is `True`, parties maintain their CPD locally, while if it is `False`, the initiator collects all CPDs, multiplying them together locally, while the others discard their local CPD. Finally, through the `mark_overlap_solved` method, parties update their `solved_overlaps` and `no_marg_nodes` fields based on the overlap feature and its parents, while updates to `node_to_cpd`, `node_to_fact`, and `node_to_nr_states` depend on the availability of a local CPD.

### SAVE (`Party.query`)

For each requested query, all parties first perform inference locally. Parties propagate any evidence present in the factors of `node_to_fact` through reduction, and store the resulting list of factors in the `party_facts` variable. Furthermore, they store in the `nodes` variable all the features present in their factors, removing query targets and evidence, as well as their `no_marg_nodes`. They, thus, run variable elimination on the factors from `party_facts`, eliminating the variables in `nodes`. The resulting factors form the intermediate results upon which the requesting party runs the final round of variable elimination, where the variables to be eliminated are all those present in the imtermediate factors that are not also query targets.

### Name Obfuscation

Note that for easier development and experimentation, names of features and their states are not obfuscated in communication between parties. To enable the mechanism, during initialization, each feature from the input Bayesian network would be assigned an obfuscated name. Similarly, feature state names would need to be assigned an obfuscated representation (e.g., by suffixing the obfuscated feature name with increasing integers). Before solving any overlaps, for each overlapping feature, involved parties would have to agree on an obfuscated name (e.g., by picking one out of their existing ones), renouncing the one they had previously chosen independently. Since a small chance of collision between names exists, in case there is no tolerance for the risk, parties can all share amongst themselves the reserved obfuscated names to allow picking alternatives when required. For all subsequent communication between parties for solving overlaps or doing inference, obfuscated feature (state) names would be used, but users would still only see the original names when interacting with the system. The following function can be used to generate obfuscated feature names:
```python
def obfuscate(rand: random.Random, prefix="_", toks=string.ascii_letters, mn_l=15, mx_l=30) -> str:
    return "".join(itertools.chain(prefix, rand.choices(toks, k=rand.randint(mn_l, mx_l))))
```

### Overlap Feature States

In the tested scenarios, overlap features always have the same set of known states across parties. Should that not be the case, multiple strategies exist for addressing the situations.

One is to bundle all states not present in all parties under a single miscellaneous state. Thus, before making other changes to their overlap CPDs, parties replace states not common to all others in the overlap with the new catch-all state, setting, for each column, its entry as the sum of the corresponding ones it subsumes.

Another is to keep working with the union of all states. Before further changes to their overlap CPDs, parties would then add any states not previously modeled. The new states could have a probability that is either a fixed (small) value or inversely proportional to the number of states in the union. To ensures that column entries would still sum up to `1`, normalization could be employed as per usual. Alternatively, existing probabilities could be replaced by their multiplication with `1` minus the sum of the introduced values, assuming the latter is less than `1` itself.

## Non-Simulated Deployment Notes

The following are some things to keep in mind for running the framework in multi-machine settings.

Assuming an implementation still in Python, usage of the [random module](https://docs.python.org/3/library/random.html) should generally be replaced with the [secrets one](https://docs.python.org/3/library/secrets.html), which purposefully does not allow seeding to allow randomness suitable for cryptographic purposes.

For the private set intersection, using a [protocol other than RAW](https://github.com/OpenMined/PSI#protocol) helps considerably reduce communication size at the cost of a a chance for false positives.

Parties involved in the network need a way to discover each other, specifically a way to broadcast their presence to all others so that the protocol can scuccesfully run in a decentralized manner.

When new parties join the network or existing parties update their local model, _CABN_ does not have to rerun from scratch. For private set intersections, only those involving new/updated parties must be rerun. Similarly, overlaps for features whose parties did not change, can be skipped during solving.

To reduce _CABN_ runtime, once all parties know their pairwise intersections and feature states are set in place, overlaps can be solved at the same time, without degrading the final result. Instead of one `tmp_vals` field, parties would have multiple ones for all the overlaps being solved at any given moment.

Different strategies can dictate when _CABN_ runs after the initial call, each with its own advantages and drawbacks. One could be to run the protocol at fixed time intervals and abort if no updates exist, making the workload more predictable. Another would be to wait until a specified number of updates happen before rerunning, responding to change faster as the chosen figure decreases, but increasing the time spent propagating updates amongst parties.

To ensure that _SAVE_ can still run while _CABN_ is processing updates, parties can have two state partitions, for the current stable representation and the newer one in the process of being finalized, respectively. Thus, once an update is fully processed, parties can simply agree to switch together to the same representation simultaneously, ensuring that there is minimal to no disruption of service in the ability to perform analysis.
