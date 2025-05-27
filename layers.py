from math import log , floor

w_value=0

## Compute binom(n,k) in big integers
## using a precomputed table if possible
def binom(n,k):
    if binoms[n][k]==0:
        value = factorials[n]//(factorials[k]*factorials[n-k])
        binoms[n][k]=value
        return value
    else:
        return binoms[n][k]

## Compute the layer-@k of hypercube of dimension @n and range [0;@m]
## Equals the coefficient of x^k in the product (1+x+x^2+... +x^{m})^n
## Requires n>1 and k<=n*m
def nb(k,m,n):
    sum=0
    for s in range(0,1+floor(k/(m+1))):
        summand = binom(n,s)*binom(k-s*(m+1)+n-1,n-1)
        if(s&1==0):
            sum += summand
        else:
            sum -= summand
    return sum

def test_precompute():
    precompute_real(10,11)
    print(nb(50,10,10))
    print(layer_sizes[50])

## Precomputing the hypercube layer sizes. Hypercube consists of
## @dimension elements each taking values from 0 to val-1
def precompute_real(dimension,val):
    global powcc
    powcc=1   #compute the size of the hypercube
    for i in range(dimension):
        powcc*=val
    global max_distance
    max_distance = (val-1)*dimension  #distance between all-0 and all-val-1
    global factorials;  #array of factorials
    factorials = [0 for i in range(max_distance+dimension+1)] #some margin
    factorials[0]=1
    for i in range(1,max_distance+dimension):
        factorials[i] = factorials[i-1]*i
    global binoms;   #double array of binoms
    binoms = [[0 for i  in range(max_distance+dimension)] for j in range(max_distance+dimension)]
    global layer_sizes
    layer_sizes =[0 for i in range(max_distance+1)]
    for i in range(max_distance+1):
        layer_sizes[i] = nb(i,val-1,dimension)
    global w_value
    w_value = val


#find layer index D such that L_{[0:D]} exceeds 2^sec_level in the hypercube [w]^v
# returns -1 if no such D
def find_optimal_layer_index(v,w,sec_level):
    #compute layer sizes
    precompute_real(v,w)
    #compute top layer size
    sum_ld = 0
    max_layer_index = v*(w-1)-1
    for D in range(0,max_layer_index):
        sum_ld += layer_sizes[D]
        if(log(sum_ld,2) >= sec_level):
            return D
    return -1

#find layer index DT such that we can get into L_DT
# after 2^grind_level attempts while sampling to L_{[0:D]}
# returns -1 if no such DT exists
def find_target_sum_index(v,w,D,grind_level):
    #compute layer sizes
    precompute_real(v,w)
    #compute top layer size
    sum_ld = 0
    for d in range(0,D+1):
        sum_ld += layer_sizes[d]
    for dt in range(0,D+1):
        if(log(sum_ld,2) <= log(layer_sizes[dt],2)+grind_level):
            return dt
    return -1


def layer_to_domain_ratio(v,w,D,T):
    target_layer = v * (w-1) - T
    precompute_real(v,w)
    sum_ld = 0
    for d in range(0,D+1):
        sum_ld += layer_sizes[d]
    return layer_sizes[target_layer] / sum_ld