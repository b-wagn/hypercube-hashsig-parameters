
SECURITY_LEVEL_CLASSICAL = 128
SECURITY_LEVEL_QUANTUM = 64
LOG_K = 12
LOG_FIELD_SIZE = 31

from collections import Counter
import math
from typing import List, Tuple
import layers

###################################################################################################
#                                      Helper Functions                                           #
###################################################################################################


def bytes_per_field_element(log_field_size: int) -> int:
    """
    Returns the number of bytes to encode a field element.
    """
    # assume k <= log p < k+1. Then 2^k <= p < 2^{k+1}
    # this means we need k+1 bits to represent a field element
    bits = math.floor(log_field_size) + 1
    bytes = math.ceil(bits / 8)
    return bytes


def field_elements_to_encode(log_field_size: int, input_len: int) -> int:
    """
    Returns the number of field elements we need if
    we want to encode a message of length input_len bits.
    I.e., it returns chi such that p^chi > 2^input_len
    """

    # The number of field elements chi is the minimum chi such that p^chi > 2^input_len
    # or equivalently, chi log p > input_len
    res = math.ceil(input_len / log_field_size)

    # we want strict inequality, so maybe we need to add one
    if input_len % log_field_size == 0:
        res += 1

    return res



###################################################################################################
#                              Functions related to tweaks                                        #
###################################################################################################


def tweak_parameters_can_fit_into_integer_bounds(log_lifetime: int, num_chains: int, chain_length: int) -> bool:
    """
    Checks that the parameters are chosen such that the tweak can be represented by
    appropriate integer bounds, e.g., level in the tree can be represented by u8.
    """

    # level in tree should be u8
    if log_lifetime + 1 > 2**8:
        return False

    # pos in level in tree should be u32
    # and epoch should be u32
    if log_lifetime > 32:
        return False

    # chain index should be u16
    if num_chains > 2**16:
        return False

    # position in chain should be u16
    if chain_length > 2**16:
        return False

    return True


def tweak_length_fe_chain_and_tree(log_field_size: int) -> int:
    """
    Determine how many field elements we need to represent our tweaks for hashing in chains and tree.
    """

    # longest tweak contains:
    #       1 Byte for domain separator,
    #       u32 for epoch,
    #       u16 for chain index,
    #       u16 for position in chain
    num_bits = 8 + 32 + 16 + 16
    return field_elements_to_encode(log_field_size, num_bits)




###################################################################################################
#                Functions and constants related to Poseidon permutation widths                   #
###################################################################################################


PERMUTATION_WIDTH_MESSAGE_HASH = 24
PERMUTATION_WIDTH_TREE_HASH = 24
PERMUTATION_WIDTH_CHAIN_HASH = 16

def round_to_valid_width(width: int) -> int:
    """
    rounds a desired permutation width up to a valid width.
    A valid width for Poseidon2 is a multiple of 4.
    """
    return math.ceil(width / 4) * 4


def permutation_width_24_enough_for_message_hash(
    log_field_size: int, parameter_len_fe: int, rand_len_fe: int
) -> bool:
    """
    Checks if permutation width 24 is enough to be used for message hashing
    """

    message_len_fe = field_elements_to_encode(log_field_size, 256)
    # tweak here is domain separator + epoch, epoch is u32
    tweak_len_fe = field_elements_to_encode(log_field_size, 32 + 8)

    min_width = round_to_valid_width(
        parameter_len_fe + tweak_len_fe + message_len_fe + rand_len_fe
    )
    return min_width <= PERMUTATION_WIDTH_MESSAGE_HASH


def permutation_width_16_enough_for_chain_hash(
    parameter_len_fe: int, tweak_len_fe: int, hash_len_fe: int
) -> bool:
    """
    Checks if permutation width 16 is enough to be used for chain hashing.
    """
    min_width = round_to_valid_width(parameter_len_fe + tweak_len_fe + hash_len_fe)
    return min_width <= PERMUTATION_WIDTH_CHAIN_HASH and hash_len_fe <= PERMUTATION_WIDTH_CHAIN_HASH


def permutation_width_24_enough_for_tree_hash(
    parameter_len_fe: int, tweak_len_fe: int, hash_len_fe: int
) -> bool:
    """
    Checks if permutation width 24 is enough to be used for tree hashing.
    """
    min_width = round_to_valid_width(parameter_len_fe + tweak_len_fe + 2 * hash_len_fe)
    return min_width <= PERMUTATION_WIDTH_TREE_HASH and hash_len_fe <= PERMUTATION_WIDTH_TREE_HASH



def permutation_widths_leaf_hash(
    log_field_size: int, security_level_classical: int, security_level_quantum: int, parameter_len_fe: int, tweak_len_fe: int, hash_len_fe: int, num_chains: int
) -> List[int]:
    """
    Returns the list of permutation widths that the leaf
    hash uses internally. The leaf hash uses the Sponge mode,
    and so multiple different invocations of the Poseidon
    permutation are used.
    """

    # first determine capacity
    capacity_lower_bound_classical = math.ceil( 2* security_level_classical / log_field_size)
    capacity_lower_bound_quantum = math.ceil(3 * security_level_quantum / log_field_size)
    capacity = max(capacity_lower_bound_classical, capacity_lower_bound_quantum)

    # initially, we use a call to PoseidonCompress with internal
    # permutation width of 24.
    widths = [24]

    # now, we loop for s iterations, with a permutation of 24 per iteration
    # So, we first need to determine s. s should be minimum such that rate * s >=
    # the length of parameter, tweak, and input.
    par_tweak_mes_len = parameter_len_fe + tweak_len_fe + num_chains * hash_len_fe
    rate = 24 - capacity
    s = math.ceil(par_tweak_mes_len / rate)
    widths += s * [24]

    return widths



###################################################################################################
#                            Functions to compute individual parameters                           #
###################################################################################################

def randomness_length_fe(log_field_size: int, log_lifetime: int, log_K: int, security_level_classical: int, security_level_quantum: int) -> int:
    """
    Determine the number of field elements we need to use for
    message hashing randomness.
    """

    # lower bounds on log|R| imposed by security bounds
    rand_len_bits_classical = math.ceil(
        security_level_classical + math.log2(5) + log_lifetime + log_K + 1
    )
    rand_len_bits_quantum = math.ceil(
        2 * (security_level_quantum + math.log2(5) + math.log2(3) + log_K)
        + log_lifetime
    )
    lower_bound_bits = max(rand_len_bits_classical, rand_len_bits_quantum)

    # round up to field elements
    return math.ceil(lower_bound_bits / log_field_size)


def parameter_length_fe(log_field_size: int, security_level_classical: int, security_level_quantum: int) -> int:
    """
    Determine the number of field elements we need to use for our public parameter
    for tweakable hashing.
    """

    # lower bounds on log|P| imposed by security bounds
    par_len_bits_classical = math.ceil(security_level_classical + math.log2(5) + 3)
    par_len_bits_quantum = math.ceil(2 * (security_level_quantum + math.log2(5) + 2) + 5)
    lower_bound_bits = max(par_len_bits_classical, par_len_bits_quantum)

    # round up to field elements
    return math.ceil(lower_bound_bits / log_field_size)


def hash_length_fe(log_field_size: int, log_lifetime: int, num_chains: int, chain_length: int, security_level_classical: int, security_level_quantum: int) -> int:
    """
    Determine the number of field elements that our tweakable hash function in chains and tree
    needs to output.
    """
    # lower bounds on log|H| imposed by security bounds
    hash_len_bits_classical = math.ceil(
        security_level_classical
        + math.log2(5)
        + 2 * math.log2(chain_length)
        + log_lifetime
        + math.log2(num_chains)
    )
    hash_len_bits_quantum = math.ceil(
        2
        * (
            security_level_quantum
            + math.log2(5)
            + 2 * math.log2(chain_length)
            + log_lifetime
            + math.log2(num_chains)
            + math.log2(12)
        )
    )
    lower_bound_bits = max(hash_len_bits_classical, hash_len_bits_quantum)

    # round up to field elements
    return math.ceil(lower_bound_bits / log_field_size)


def final_layer_of_domain(num_chains: int, chain_length: int, security_level_classical: int, security_level_quantum: int) -> int:
    """
    Determines the part of the hypercube into which we map. That is D in ell_{[0:D]}.
    This is picked such that ell_{[0:D]} (size of this part of the hypercube) is large enough.
    """
    level_classical = security_level_classical + math.log2(5) + 1
    level_quantum = 2 * (security_level_quantum + math.log2(5) + 1) + 3
    level = max(level_classical, level_quantum)
    return layers.find_optimal_layer_index(num_chains, chain_length, level)

def pick_target_sum(num_chains: int, chain_length: int, security_level_classical: int, security_level_quantum: int) -> int:
    """
    Picks a target sum that is a good fit.
    """
    final_layer = final_layer_of_domain(num_chains, chain_length, security_level_classical, security_level_quantum) - 4
    return num_chains * (chain_length - 1) - final_layer


###################################################################################################
#                                    Estimated Correctness Error                                  #
###################################################################################################

def correctness_error_per_trial(num_chains: int, chain_length: int, target_sum: int, domain_layer: int) -> float:
    return 1.0 - layers.layer_to_domain_ratio(num_chains, chain_length, domain_layer, target_sum)

def expected_number_of_trials(num_chains: int, chain_length: int, target_sum: int, domain_layer: int) -> float:
    return 1.0 / (1.0 - correctness_error_per_trial(num_chains, chain_length, target_sum, domain_layer))

def correctness_error_for_k_trials(num_chains: int, chain_length: int, target_sum: int, domain_layer: int, log_k: int) -> float:
    K = 2 ** log_k
    return correctness_error_per_trial(num_chains, chain_length, target_sum, domain_layer) ** K


###################################################################################################
#                                       Signature size                                            #
###################################################################################################

def merkle_path_size_fe(log_lifetime: int, hash_len_fe: int) -> int:
    """
    Returns the size of a Merkle path in field elements,
    assuming hash_len_fe is given infield elements.
    """
    num_hashes = log_lifetime
    return num_hashes * hash_len_fe


def signature_size_fe(
    log_lifetime: int, hash_len_fe: int, rand_len_fe: int, num_chains: int,
) -> int:
    """
    Returns the size of a signature (in field elements), given the parameters.
    """
    signature_size = 0

    # The signature contains randomness for incomparable encoding / message hash
    signature_size += rand_len_fe

    # The signature contains the Merkle path
    signature_size += merkle_path_size_fe(log_lifetime, hash_len_fe)

    # For each chain, the signature contains one hash
    # There is one chain per chunk
    signature_size += num_chains * hash_len_fe

    return signature_size




###################################################################################################
#                                       Verifier hashing                                          #
###################################################################################################

def verifier_hashing(
    log_lifetime: int, num_chains: int, chain_length: int, target_sum: int, tweak_len_fe: int, parameter_len_fe: int, hash_len_fe: int
) -> List[Tuple[int, int]]:
    """
    Returns the hash complexity of verification.

    Note: the output is a list of pairs (width, count), and each such pair
    indicates that the Poseidon permutation of width `width` is called `count`
    many times.
    """
    hashing = []

    # Encode the message, which involves two permutations of width 24
    hashing += [PERMUTATION_WIDTH_MESSAGE_HASH, PERMUTATION_WIDTH_MESSAGE_HASH]

    # For the chains: determine how many steps are needed in total
    chain_steps_signer = target_sum
    chain_steps_total = num_chains * (chain_length - 1)
    chain_steps_verifier = chain_steps_total - chain_steps_signer

    # For each step, we do a Poseidon permutation of width 16
    hashing += chain_steps_verifier * [PERMUTATION_WIDTH_CHAIN_HASH]

    # Now, we hash the chain ends to get the leaf
    hashing += permutation_widths_leaf_hash(LOG_FIELD_SIZE, SECURITY_LEVEL_CLASSICAL, SECURITY_LEVEL_QUANTUM, parameter_len_fe, tweak_len_fe, hash_len_fe, num_chains)

    # We verify the Merkle path
    hashing += log_lifetime * [PERMUTATION_WIDTH_TREE_HASH]

    # Now, hashing contains all invocations separately, but we want to
    # group them (compute a histogram in some sense)
    return list(Counter(hashing).items())



###################################################################################################
#                  Summary: compute everything given num_chains and chain_length                  #
###################################################################################################

def compute_parameters(log_lifetime: int, num_chains: int, chain_length: int):
    """
    outputs parameters like hash output length, randomness length, etc
    and the resulting efficiency (signature size, verifier hashing)
    outputs the result as a dictionary.
    """

    # determine how many field elements we need for randomness
    rand_len_fe = randomness_length_fe(LOG_FIELD_SIZE, log_lifetime, LOG_K, SECURITY_LEVEL_CLASSICAL, SECURITY_LEVEL_QUANTUM)

    # determine how large our domain needs to be (subset of the hypercube)
    domain_layer = final_layer_of_domain(num_chains, chain_length, SECURITY_LEVEL_CLASSICAL, SECURITY_LEVEL_QUANTUM)
    assert domain_layer >= 0, "Cannot find a suitable domain with these parameters"

    # determine the target sum
    target_sum = pick_target_sum(num_chains, chain_length, SECURITY_LEVEL_CLASSICAL, SECURITY_LEVEL_QUANTUM)

    # determine how many field elements we need for our parameter
    par_len_fe = parameter_length_fe(LOG_FIELD_SIZE, SECURITY_LEVEL_CLASSICAL, SECURITY_LEVEL_QUANTUM)

    # determine how many field elements our
    # tweakable hash in chains and tree needs to output
    hash_len_fe = hash_length_fe(LOG_FIELD_SIZE, log_lifetime, num_chains, chain_length, SECURITY_LEVEL_CLASSICAL, SECURITY_LEVEL_QUANTUM)

    # assert that the input lengths for the tweakable hash can hold
    # all possible inputs (including their tweaks and parameters)
    # and that parameters make sense to encode tweaks
    tweak_len_fe = tweak_length_fe_chain_and_tree(LOG_FIELD_SIZE)
    assert tweak_parameters_can_fit_into_integer_bounds(log_lifetime, num_chains, chain_length), "Cannot encode tweaks by appropriate integers"
    assert permutation_width_16_enough_for_chain_hash(par_len_fe, tweak_len_fe, hash_len_fe), "Permutation width 16 not enough for chain hash"
    assert permutation_width_24_enough_for_tree_hash(par_len_fe, tweak_len_fe, hash_len_fe), "Permutation width 24 not enough for tree hash"
    assert permutation_width_24_enough_for_message_hash(LOG_FIELD_SIZE, par_len_fe, rand_len_fe), "Permutation width 24 not enough for message hash"

    # Now that we have all parameters, we determine the efficiency

    # determine amount of verification hashing
    hashing = verifier_hashing(log_lifetime, num_chains, chain_length, target_sum, tweak_len_fe, par_len_fe, hash_len_fe)

    # determine signature size
    sig_size_fe = signature_size_fe(log_lifetime, hash_len_fe, rand_len_fe, num_chains)
    sig_size_kilobytes = bytes_per_field_element(LOG_FIELD_SIZE) * sig_size_fe / 1024

    # determine correctness error per trial, expected nr of trials, and full correctness error
    corr_error_per_trial = correctness_error_per_trial(num_chains, chain_length, target_sum, domain_layer)
    expected_num_trials = expected_number_of_trials(num_chains, chain_length, target_sum, domain_layer)
    corr_error = correctness_error_for_k_trials(num_chains, chain_length, target_sum, domain_layer, LOG_K)

    # return the result as a dict
    return {
        'rand_len_fe': rand_len_fe,
        'domain_layer': domain_layer,
        'target_sum': target_sum,
        'par_len_fe': par_len_fe,
        'hash_len_fe': hash_len_fe,
        'tweak_len_fe': tweak_len_fe,
        'hashing': hashing,
        'sig_size_fe': sig_size_fe,
        'sig_size_kilobytes': sig_size_kilobytes,
        'corr_error_per_trial': corr_error_per_trial,
        'expected_num_trials': expected_num_trials,
        'corr_error': corr_error
    }


###################################################################################################
#                                        User interface                                           #
###################################################################################################


import argparse
import pprint

def generate_rust_code(params, log_lifetime, num_chains, chain_length):
    rust_template = f"""\
const LOG_LIFETIME: usize = {log_lifetime};

const DIMENSION: usize = {num_chains};
const BASE: usize = {chain_length};
const FINAL_LAYER: usize = {params['domain_layer']};
const TARGET_SUM: usize = {params['target_sum']};

const PARAMETER_LEN: usize = {params['par_len_fe']};
const TWEAK_LEN_FE: usize = {params['tweak_len_fe']};
const MSG_LEN_FE: usize = 9;
const RAND_LEN_FE: usize = {params['rand_len_fe']};
const HASH_LEN_FE: usize = {params['hash_len_fe']};

const CAPACITY: usize = 9;

const POSEIDON_INVOCATIONS: usize = 2;
const POS_OUTPUT_LEN_FE: usize = POSEIDON_INVOCATIONS * 24;

type MH = TopLevelPoseidonMessageHash<
    POS_OUTPUT_LEN_FE,
    DIMENSION,
    BASE,
    FINAL_LAYER,
    TWEAK_LEN_FE,
    MSG_LEN_FE,
    PARAMETER_LEN,
    RAND_LEN_FE,
>;
type TH = PoseidonTweakHash<PARAMETER_LEN, HASH_LEN_FE, TWEAK_LEN_FE, CAPACITY, DIMENSION>;
type PRF = ShakePRFtoF<HASH_LEN_FE>;
type IE = TargetSumEncoding<MH, TARGET_SUM>;

pub type SIGTopLevelTargetSumLifetime{log_lifetime}Dim{num_chains}Base{chain_length} =
    GeneralizedXMSSSignatureScheme<PRF, IE, TH, LOG_LIFETIME>;

#[cfg(test)]
mod test {{

    #[cfg(feature = "slow-tests")]
    use crate::signature::test_templates::_test_signature_scheme_correctness;

    #[test]
    pub fn test_internal_consistency() {{
        SIGTopLevelTargetSumLifetime{log_lifetime}Dim{num_chains}Base{chain_length}::internal_consistency_check();
    }}

    #[test]
    #[cfg(feature = "slow-tests")]
    pub fn test_correctness() {{
        _test_signature_scheme_correctness::<SIGTopLevelTargetSumLifetime{log_lifetime}Dim{num_chains}Base{chain_length}>(213);
        _test_signature_scheme_correctness::<SIGTopLevelTargetSumLifetime{log_lifetime}Dim{num_chains}Base{chain_length}>(4);
    }}
}}"""
    return rust_template


def main():
    parser = argparse.ArgumentParser(description="Compute signature scheme parameters.")
    parser.add_argument("log_lifetime", type=int, help="Log_2 of the key lifetime")
    parser.add_argument("num_chains", type=int, help="Number of chains")
    parser.add_argument("chain_length", type=int, help="Length of each chain")
    parser.add_argument("--rustcode", action="store_true", help="Output Rust code with computed parameters")

    args = parser.parse_args()

    # Call the function with the provided arguments
    result = compute_parameters(args.log_lifetime, args.num_chains, args.chain_length)

    if args.rustcode:
        rust_code = generate_rust_code(result, args.log_lifetime, args.num_chains, args.chain_length)
        print("\nGenerated Rust Code:\n")
        print(rust_code)
    else:
        print("\nComputed Parameters:")
        pprint.pprint(result, sort_dicts=False)


if __name__ == "__main__":
    main()
