import os
cimport cython
import numpy as np
cimport numpy as np
from functools import partial
from .pymmh3 import hash128, hash64
from .parallel import parallel_process

class MinHash(object):
    def __init__(self, numband, bandwidth):
        self.numhash = numband * bandwidth
        self.numband = numband
        self.bandwidth = bandwidth
    def fit(self, csc_matrix, njobs=os.cpu_count()):
        self.csc_matrix = csc_matrix
        self.signature_matrix = generate_signature_matrix(self, csc_matrix, njobs)
        self.bucket_list = generate_bucket_list(self, self.signature_matrix, njobs)
    def predict(self, col):
        """
        This function returns indices whose corresponding colomn vectors are similar in the sense of Jaccard similarity. 
        """
        candidates = set()
        for idx, band in enumerate(range(0, self.numhash, self.bandwidth)):
            hash_ = hash128(
                self.signature_matrix[band:(band+self.bandwidth), col], band
            )
            for item in self.bucket_list[idx][hash_]:
                candidates.add(item)
        return candidates
    # def gen_bucket_list(self):
    #     self.bucket_list = []
    #     for band in range(0, self.numhash, self.bandwidth):
    #         bucket = dict()
    #         for col in range(self.signature_matrix.shape[1]):
    #             hash_ = hash128(self.signature_matrix[band:(band+bandwidth), col], band)
    #             if not hash_ in bucket:
    #                 bucket[hash_] = set()
    #             bucket[hash_].add(col)
    #         if len(bucket) > 0:
    #             self.bucket_list.append(bucket)



def minhash(words, seed=0):
    current_min = np.inf
    minhash_word = None
    for word in words:
        hash_ = hash128(word, seed)
        if hash_ < current_min:
            minhash_word = word
            current_min = hash_
    return minhash_word


def generate_signature_matrix(minhash, csc_mat, njobs=-1):
    if njobs == 1:
        return generate_signature_matrix_cpu_single(
            minhash.numhash, minhash.numband, minhash.bandwidth, csc_mat.astype(np.uint32))
    else:
        return generate_signature_matrix_cpu_multi(
            minhash.numhash, minhash.numband, minhash.bandwidth, csc_mat.astype(np.uint32), njobs)


def generate_signature_matrix_cpu_single(numhash, numband, bandwidth, csc_mat):
    signature_matrix = np.zeros((numhash, csc_mat.shape[1]), dtype=np.uint32)
    for row in range(numhash):
        for col in range(csc_mat.shape[1]):
            idsets = csc_mat[:, col].indices
            if len(idsets) > 0:
                signature_matrix[row, col] = minhash(idsets, seed=row)
            else:
                signature_matrix[row, col] = hash64(3511 * col, seed=row)[0]
    return signature_matrix

def _generate_signature_vec(numhash, numband, bandwidth, csc_mat, col):
    signature_vec = np.zeros(numhash, dtype=np.uint32)
    idsets = csc_mat[:, col].indices
    for row in range(numhash):
        if len(idsets) > 0:
            signature_vec[row] = minhash(idsets, seed=row)
        else:
            signature_vec[row] = hash64(3511 * col, seed=row)[0]
    return col, signature_vec

def generate_signature_matrix_cpu_multi(numhash, numband, bandwidth, csc_mat, njobs):
    njobs = os.cpu_count() if njobs == -1 else njobs
    signature_matrix = np.zeros((numhash, csc_mat.shape[1]), dtype=np.uint32)
    worker = partial(
        _generate_signature_vec,
        numhash=numhash,
        numband=numband,
        bandwidth=bandwidth,
        csc_mat=csc_mat,
    )
    worker.__name__ = "generate_signature_matrix_cpu_multi"
    args = [{
        "col": col
    } for col in range(csc_mat.shape[1])]
    results = parallel_process(args, worker, njobs, use_kwargs = True)
    for col, sigvec in results:
        signature_matrix[:, col] = sigvec
    return signature_matrix

def generate_bucket_list(minhash, signature_matrix, njobs):
    """ multiprocessing version is not implemented yet """
    if njobs == 1:
        return generate_bucket_list_single(
            minhash.numhash, minhash.numband, minhash.bandwidth, signature_matrix)
    else:
        return generate_bucket_list_single(
            minhash.numhash, minhash.numband, minhash.bandwidth, signature_matrix)

def generate_bucket_list_single(numhash, numband, bandwidth, signature_matrix):
    bucket_list = []
    for band in range(0, numhash, bandwidth):
        bucket = dict()
        for col in range(signature_matrix.shape[1]):
            hash_ = hash128(signature_matrix[band:(band+bandwidth), col], band)
            if not hash_ in bucket:
                bucket[hash_] = set()
            bucket[hash_].add(col)
        if len(bucket) > 0:
            bucket_list.append(bucket)
    return bucket_list

def find_similar(numhash, numband, bandwidth, signature_matrix, bucket_list, col):
    candidates = set()
    for idx, band in enumerate(range(0, numhash, bandwidth)):
        hash_ = hash128(signature_matrix[band:(band+bandwidth), col], band)
        for item in bucket_list[idx][hash_]:
            candidates.add(item)
    return candidates


