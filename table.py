from hypercube import compute_parameters
from tabulate import tabulate
range_num_chains = range(80, 30, -4)
range_chain_length = range(20, 2, -2)

curr_sig_size = 500000000

headers = ["Number of Chains", "Chain Length", "Target Sum", "Verifier Hashing", "Signature Size [KiB]"]
table = []

for num_chains in range_num_chains:
    for chain_length in range_chain_length:
        try:
            parameters = compute_parameters(num_chains, chain_length)
            row = [
                num_chains,
                chain_length,
                parameters['target_sum'],
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

print(
    tabulate(
        table, headers=headers, tablefmt="pretty"
    )
)
