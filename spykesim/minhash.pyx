cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp
cimport openmp
from libc.math cimport log
from cython.parallel cimport prange
from cython.parallel cimport parallel

DBL = np.double
ctypedef np.double_t DBL_C
INT = np.int
ctypedef np.int_t INT_C

class MinHash(object):
    def __init__(self, numband, bandwidth):
        self.numhash = numband * bandwidth
        self.numband = numband
        self.bandwidth = bandwidth

def hash_fn(key, seed = 0):
    a = 63689
    b = 378551
    return (a * b * (key + seed + 1)) & 0x7FFFFFFF

def minhash(words, seed = 0):
    current_min = np.inf
    minhash_word = None
    for word in words:
        hash_ = hash_fn(word, seed)
        if hash_ < current_min:
            minhash_word = word
            current_min = hash_
    return minhash_word     
def generate_signature_matrix(minhash, data, mode = "cpu"):
    if mode == "cpu": 
        return _generate_signature_matrix_cpu(
            minhash.numhash, minhash.numband, minhash.bandwidth, data)
    #elif mode == "gpu": 
    #    return _generate_signature_matrix_gpu(
    #        minhash.numhash, minhash.numband, minhash.bandwidth, data)
    else: raise RuntimeError('Option must be eather cpu or gpu')
    
def _generate_signature_matrix_cpu(numhash, numband, bandwidth, data):
    signature_matrix = np.zeros((numhash, data.shape[1]), dtype = np.uint32)
    for row in range(numhash):
        for col in range(data.shape[1]):
            idsets = np.where(data[:, col] >= 1)[0]
            if len(idsets) > 0: 
                signature_matrix[row, col] = minhash(idsets, seed = row)
            else:
                signature_matrix[row, col] = hash_fn(3511 * col, seed = row)
    return signature_matrix

