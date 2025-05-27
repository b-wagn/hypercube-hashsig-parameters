# Parameter estimations for XMSS-like signatures using novel encodings
This repository contains preliminary script to set parameters for hash-based signatures that instantiate XMSS as in [this paper](https://eprint.iacr.org/2025/055.pdf), but using novel encodings which are partially inspired by [this paper](https://eprint.iacr.org/2025/889.pdf).

The idea is to replace the incomparable encodings used [here](https://eprint.iacr.org/2025/055.pdf) with incomparable encodings mapping into the top layers of a larger hypercube, as suggested [here](https://eprint.iacr.org/2025/889.pdf).

Disclaimer: This not meant to be used in production, parameters have not been audited and are just estimates.

## Usage
Run the script with
```
python3 hypercube.py <number of chains> <length of chains>
```
to get parameter estimates. Some combinations of number of chains and length of chains do not lead to any secure setting of parameters.
In this case, the script fails with an assertion error.


## License
Apache License, Version 2.0