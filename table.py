from hypercube import compute_parameters
from tabulate import tabulate
import argparse

# Argument parser for configurable input
parser = argparse.ArgumentParser(description="Compute optimal hypercube parameters.")
parser.add_argument(
    "--log_lifetime",
    type=int,
    default=32,
    help="Base-2 logarithm of lifetime (default: 32)"
)
parser.add_argument(
    "--max_expected_tries",
    type=int,
    default=30,
    help="Maximum expected number of tries for signer (default: 30)"
)
args = parser.parse_args()

log_lifetime = args.log_lifetime
max_expected_tries = args.max_expected_tries

range_num_chains = range(80, 30, -4)
range_chain_length = range(20, 2, -2)

curr_sig_size = 500000000

headers = ["Number of Chains", "Chain Length", "Target Sum", "Expected Nr of Tries", "Verifier Hashing", "Signature Size [KiB]"]
table = []

for num_chains in range_num_chains:
    for chain_length in range_chain_length:
        try:
            parameters = compute_parameters(log_lifetime, num_chains, chain_length, max_expected_tries)
            row = [
                num_chains,
                chain_length,
                parameters['target_sum'],
                parameters['expected_num_trials'],
                parameters['hashing'],
                parameters['sig_size_kilobytes']
            ]

            if parameters['sig_size_kilobytes'] > curr_sig_size:
                continue

            curr_sig_size =  parameters['sig_size_kilobytes']
            table.append(row)
        except AssertionError:
            # Skip this combination silently
            continue
        except Exception as e:
            # Optionally log unexpected exceptions
            print(f"Error with num_chains={num_chains}, chain_length={chain_length}: {e}")
            continue

print("Table for lifetime 2^" + str(log_lifetime))

print(
    tabulate(
        table, headers=headers, tablefmt="pretty"
    )
)
