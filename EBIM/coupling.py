import numpy as np
from scipy.stats import entropy
import math
import warnings



def merge_columns(matrix, indices):
    # Calculate the sum of columns corresponding to the indices
    summed_column = np.sum(matrix[:, indices], axis=1)

    # Remove the original columns specified by the indices
    matrix = np.delete(matrix, indices, axis=1)

    # Append the summed column to the right of the matrix
    return np.hstack((matrix, summed_column[:, np.newaxis]))


def merge_two_columns(matrix, c1, c2):
    matrix = matrix.copy()
    matrix[:, c1] += matrix[:, c2]
    return np.delete(matrix, c2, axis=1)


def merge_two_columns_(matrix, c1, c2):
    matrix[:, c1] += matrix[:, c2]
    
    sel = np.full(matrix.shape[1], True)
    sel[c2] = False
    matrix = matrix[:, sel]
    return matrix 


def delta_H(p, q):
    if p == 0 or q == 0:
        return 0
    return p * np.log2(1 + q / p) + q * np.log2(1 + p / q)


def search_functional_mapping(Px, R):
    Pxt = np.diag(Px)
    Pxt = np.asfortranarray(Pxt)
    
    # if Px has zeros, remove from t:    
    Pxt = Pxt[:, Pxt.sum(axis=0)>0]
    
    # if R >= H(x) return Px diag:
    if R >= entropy(Px, base=2):
        return Pxt
    
    for _ in range(1, len(Px)):
        Pt = Pxt.sum(axis=0)
        Ht = entropy(Pt, base=2)
        
        # merging smallests
        two_smallest_cols = np.argpartition(Pt, 1)[:2]
        Is = Ht - delta_H(*Pt[two_smallest_cols])
        if Is <= R:
            Ps = merge_two_columns_(Pxt, *two_smallest_cols)
            return Ps
        
        # merging largests
        two_largest_cols = np.argpartition(Pxt.sum(axis=0), -2)[-2:]
        Il = Ht - delta_H(*Pt[two_largest_cols])
        Pl = merge_two_columns_(Pxt, *two_largest_cols)
        if Il <= R:
            return Pl
        else:
            Pxt = Pl
    return Pxt
            
def scalar_quant(Px, R):
    n = len(Px)

    # partition_size = math.ceil(n / (2**R))
    # m = math.ceil(n / partition_size)

    # H(T) <= log(m) & m <= 2^R => H(T)<=R
    m = math.floor(2**R) 
    partition_size = math.ceil(n / m)
    
    Pxt = np.zeros((n, m))
    
    for i, start in enumerate(range(0, n, partition_size)):
        partition = slice(start, start + partition_size)
        Pxt[partition, i] = Px[partition]
    
    return Pxt
        
            
def min_ent_coupling(Px, Py):
    Px = Px.copy()
    Py = Py.copy()
    
    Pxy = np.zeros((Px.size, Py.size))
    
    while Px.sum() > 1e-8:
        x, y = Px.argmax(), Py.argmax()
        p = np.minimum(Px[x], Py[y])
        Px[x] -= p
        Py[y] -= p
        Pxy[x, y] = p
    
    return Pxy
    
    
def bottlenecked_min_ent_coupling(Px, Py, R, comp_method='search_function'):
    if comp_method == 'search_function':
        Pxt = search_functional_mapping(Px, R)
    elif comp_method == 'scalar_quant':
        Pxt = scalar_quant(Px, R)
    else:
        raise ValueError('invalid comp_method.')
        
    Pxt = Pxt[:, Pxt.sum(axis=0)>0]
    
    Pt = Pxt.sum(axis=0)
    Pty = min_ent_coupling(Pt, Py)
    
    # p(x, y) = sum_t[ p(x, t) * p(y|t) ]
    return Pxt @ (Pty / Pt[:, None])


def mutual_info(Pxy):
    return (
        entropy(Pxy.sum(0), base=2)
        + entropy(Pxy.sum(1), base=2)
        - entropy(Pxy.ravel(), base=2)
    )