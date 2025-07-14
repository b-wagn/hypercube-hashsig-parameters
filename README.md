# Parameter estimations for XMSS-like signatures using novel encodings
This repository contains *preliminary* scripts to set parameters for hash-based signatures that instantiate XMSS as in [this paper](https://eprint.iacr.org/2025/055.pdf), but using novel encodings which are inspired by [this paper](https://eprint.iacr.org/2025/889.pdf).

The idea is to replace the incomparable encodings used [here](https://eprint.iacr.org/2025/055.pdf) with incomparable encodings mapping into the top layers of a larger hypercube, as suggested [here](https://eprint.iacr.org/2025/889.pdf).

Disclaimer: This not meant to be used in production, parameters have not been audited and are just estimates.

## Generating a Table
You can generate a table with some interesting combinations of parameters for a given key lifetime via
```
python3 table.py --log_lifetime <log2 of key lifetime> --max_expected_tries <maximum expected number of tries>
```
Note that it may take a few seconds to generate the table.
Here, *maximum expected number of tries* refers to the expected number of resampling attempts that we allow in the signer.
If we increase it, we can potentially increase the target sum and reduce verifier hashing, at the cost of less efficient signing.

## Parameters for Specific Combinations
Run the script with
```
python3 hypercube.py <log2 of key lifetime> <number of chains> <length of chains> <maximum expected number of tries>
```
to get parameter estimates.

Some combinations of number of chains and length of chains do not lead to any secure setting of parameters.
In this case, the script fails with an assertion error.

Interesting combinations are 48 chains of length 10, or 64 chains of length 8.

## License
Apache License, Version 2.0
