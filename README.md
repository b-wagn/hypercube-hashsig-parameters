# Parameter estimations for XMSS-like signatures using novel encodings
This repository contains preliminary script to set parameters for hash-based signatures that instantiate XMSS as in [this paper](https://eprint.iacr.org/2025/055.pdf), but using novel encodings which are partially inspired by [this paper](https://eprint.iacr.org/2025/889.pdf).

The idea is to replace the incomparable encodings used [here](https://eprint.iacr.org/2025/055.pdf) with incomparable encodings mapping into the top layers of a larger hypercube, as suggested [here](https://eprint.iacr.org/2025/889.pdf).

Disclaimer: This not meant to be used in production, parameters have not been audited and are just estimates.

## Generating a Table
You can generate a table with some interesting combinations of parameters for key lifetime `2^26` via
```
python3 table.py
```
Note that it may take a few seconds to generate the table.

## Parameters for Specific Combinations
Run the script with
```
python3 hypercube.py <log2 of key lifetime> <number of chains> <length of chains>
```
to get parameter estimates. Some combinations of number of chains and length of chains do not lead to any secure setting of parameters.
In this case, the script fails with an assertion error.

Interesting combinations are 50 chains of length 8, or 78 chains of length 4.

## License
Apache License, Version 2.0