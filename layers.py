from math import log, floor

w_value = 0


def binom(n, k):
    """
    Compute binom(n, k) (n choose k) using big integers.
    Uses a precomputed table if available.
    """
    if binoms[n][k] == 0:
        value = factorials[n] // (factorials[k] * factorials[n - k])
        binoms[n][k] = value
        return value
    else:
        return binoms[n][k]


def nb(k, m, n):
    """
    Compute the number of integer vectors of dimension n,
    with entries in [0, m], that sum to k.

    This is equivalent to computing the coefficient of x^k
    in the expansion of (1 + x + x^2 + ... + x^m)^n.

    Requires: n > 1 and k <= n * m
    """
    sum = 0
    for s in range(0, 1 + floor(k / (m + 1))):
        summand = binom(n, s) * binom(k - s * (m + 1) + n - 1, n - 1)
        if s % 2 == 0:
            sum += summand
        else:
            sum -= summand
    return sum


def test_nb():
    # number of vectors with two entries in {0,1} that sum to 0 should be 1
    # number of vectors with two entries in {0,1} that sum to 1 should be 2
    # number of vectors with two entries in {0,1} that sum to 2 should be 2
    # number of vectors with two entries in {0,1} that sum to 3 should be 1
    precompute_real(3, 2)
    assert nb(0, 1, 3) == 1
    assert nb(1, 1, 3) == 2
    assert nb(2, 1, 3) == 2
    assert nb(3, 1, 3) == 1


def test_precompute():
    precompute_real(10, 11)
    print(nb(50, 10, 10))
    print(layer_sizes[50])


## Precomputing binomial coefficients, factorials and hypercube layer sizes. Hypercube consists of
## @dimension elements each taking values from 0 to val-1
def precompute_real(dimension, val):
    global powcc
    powcc = 1  # compute the size of the hypercube
    for i in range(dimension):
        powcc *= val
    global max_distance
    max_distance = (val - 1) * dimension  # distance between all-0 and all-val-1
    global factorials
    # array of factorials
    factorials = [0 for i in range(max_distance + dimension + 1)]  # some margin
    factorials[0] = 1
    for i in range(1, max_distance + dimension):
        factorials[i] = factorials[i - 1] * i
    global binoms
    # double array of binoms
    binoms = [
        [0 for i in range(max_distance + dimension)]
        for j in range(max_distance + dimension)
    ]
    global layer_sizes
    layer_sizes = [0 for i in range(max_distance + 1)]
    for i in range(max_distance + 1):
        layer_sizes[i] = nb(i, val - 1, dimension)
    global w_value
    w_value = val


# find layer index D such that L_{[0:D]} exceeds 2^sec_level in the hypercube [w]^v
# returns -1 if no such D
def find_optimal_layer_index(v, w, sec_level):
    # compute layer sizes
    precompute_real(v, w)
    # compute top layer size
    sum_ld = 0
    max_layer_index = v * (w - 1) - 1
    for D in range(0, max_layer_index):
        sum_ld += layer_sizes[D]
        if log(sum_ld, 2) >= sec_level:
            return D
    return -1


def layer_to_domain_ratio(v, w, D, T):
    target_layer = v * (w - 1) - T
    precompute_real(v, w)
    sum_ld = 0
    for d in range(0, D + 1):
        sum_ld += layer_sizes[d]
    return layer_sizes[target_layer] / sum_ld
