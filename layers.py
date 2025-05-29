from math import log, floor

def binom(n: int, k: int) -> int:
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


def nb(k: int, m: int, n: int) -> int:
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



def precompute_real(v: int, w: int):
    """
    Precomputing binomial coefficients, factorials, and hypercube layer sizes.
    The hypercube is [0,w-1]^v.
    """

    global max_distance
    max_distance = (w - 1) * v  # distance between all-0 and all-val-1

    global factorials
    factorials = [0 for i in range(max_distance + v + 1)]
    factorials[0] = 1
    for i in range(1, max_distance + v):
        factorials[i] = factorials[i - 1] * i

    global binoms
    binoms = [
        [0 for i in range(max_distance + v)]
        for j in range(max_distance + v)
    ]

    global layer_sizes
    layer_sizes = [0 for i in range(max_distance + 1)]
    for i in range(max_distance + 1):
        layer_sizes[i] = nb(i, w - 1, v)




def find_optimal_layer_index(v: int, w: int, sec_level: int) -> int:
    """
    In the hypercube [0,w-1]^v, find layer index D such
    that L_{[0:D]} exceeds 2^sec_level in the hypercube [w]^v
    returns -1 if no such D
    """

    # compute layer sizes
    precompute_real(v, w)
    # compute top layer size
    sum_ld = 0
    max_layer_index = v * (w - 1)
    for D in range(0, max_layer_index):
        sum_ld += layer_sizes[D]
        if log(sum_ld, 2) >= sec_level:
            return D
    return -1


def layer_to_domain_ratio(v: int, w: int, D: int, T: int) -> int:
    """
    Assumes that target sum is T and determines the relative size of the layer with that
    target sum within the subset of the hypercube consisting of layers 0 to D.
    """
    target_layer = v * (w - 1) - T

    precompute_real(v, w)

    sum_ld = 0
    for d in range(0, D + 1):
        sum_ld += layer_sizes[d]

    return layer_sizes[target_layer] / sum_ld

def initialize_all_layer_sizes(w,v_max):
    """
    compute all layer sizes in hypercubes [w]^v with v between 1 and v_max
    store in global variable all_layer_sizes_array
    """
    global all_layer_sizes_array
    all_layer_sizes_array = [[]]
    for v in range(1,v_max+1):
        precompute_real(v,w)
        all_layer_sizes_array.append(layer_sizes)

def map_to_vertex(w,v,d,x):
    """
    maps a number 0<=x < layer_size(v,d) to an element in layer d
    """
    initialize_all_layer_sizes(w,v) #this can be called once per all calls to this function with the same (w,v)
    x_curr = x
    assert(x < all_layer_sizes_array[v][d])
    out=[]
    d_curr = d
    for i in range(1,v):
        ji=-1
        for j in range (max(0,d_curr-(w-1)*(v-i)),min(w,d_curr+1)):
            if(x_curr >= all_layer_sizes_array[v-i][d_curr-j]):
                x_curr -= all_layer_sizes_array[v-i][d_curr-j]
            else:
                ji=j
                break
        assert(ji>-1)
        assert(ji<w)
        ai = w-ji
        out.append(ai)
        d_curr -= w-ai
    assert(x_curr+d_curr<w)
    out.append(w-x_curr-d_curr)
    return out

def map_to_integer(w,v,d,a):
    """
    maps an element in layer d to number 0<=x < layer_size(v,d)
    """
    initialize_all_layer_sizes(w,v)
    assert(len(a)==v)
    x_curr = 0
    d_curr = w-a[v-1]
    for i in range(v-1,0,-1):
        ji = w-a[i-1]
        d_curr += ji
        for j in range(max(0,d_curr-(w-1)*(v-i)),ji):
            x_curr += all_layer_sizes_array[v-i][d_curr-j]
    assert(d_curr==d)
    return x_curr

def test_maps():
    w=4
    v=8
    d=20
    precompute_real(v,w)
    for x in range(layer_sizes[d]):
        a = map_to_vertex(w,v,d,x)
        y = map_to_integer(w,v,d,a)
        assert(x==y)

def test_nb():
    # number of vectors with two entries in {0,1} that sum to 0 should be 1
    # number of vectors with two entries in {0,1} that sum to 1 should be 3
    # number of vectors with two entries in {0,1} that sum to 2 should be 3
    # number of vectors with two entries in {0,1} that sum to 3 should be 1
    precompute_real(3, 2)
    assert nb(0, 1, 3) == 1
    assert nb(1, 1, 3) == 3
    assert nb(2, 1, 3) == 3
    assert nb(3, 1, 3) == 1

    # number of vectors with five entries in {0,1,2,3} that sum to 6 should be 135
    # number of vectors with five entries in {0,1,2,3} that sum to 12 should be 35
    # number of vectors with five entries in {0,1,2,3} that sum to 2 should be 15
    precompute_real(4, 5)
    assert nb(6, 3, 5) == 135
    assert nb(12, 3, 5) == 35
    assert nb(2, 3, 5) == 15


def test_precompute():
    precompute_real(10, 11)
    print(nb(50, 10, 10))
    print(layer_sizes[50])

test_maps()