from unittest import TestCase
import os
from nose.tools import ok_, eq_
from ..spykesim import minhash
import numpy as np
from scipy.sparse import csc_matrix

def jaccard(csc_mat, col1, col2):
    bvec1 = (csc_mat[:, col1] >= 1).toarray()
    bvec2 = (csc_mat[:, col2] >= 1).toarray()
    intersection = (bvec1 * bvec2).sum()
    union = (bvec1 + bvec2).sum()
    return intersection / union

class SimmatTestCase(TestCase):
    def setUp(self):
        a = np.zeros((20, 10), dtype=np.int)
        a[1:6, 0] = 1
        a[1:7, 1] = 1
        a[1:4, 2] = 1
        a[6:7, 3] = 1
        a[:, 5] = 1
        a[:, 6] = 1
        self.b = csc_matrix(a)
    def test_sigmat(self):
        numband = 20
        bandwidth = 10
        numhash = numband * bandwidth
        sigmat = minhash.generate_signature_matrix_cpu_single(numhash, numband, bandwidth, self.b)
        sigmat2 = minhash.generate_signature_matrix_cpu_multi(numhash, numband, bandwidth, self.b, 3)
        np.testing.assert_equal(sigmat, sigmat2)
    def test_minhash(self):
        numband = 20
        bandwidth = 10
        numhash = numband * bandwidth
        mh = minhash.MinHash(numband, bandwidth)
        mh.fit(self.b)
        eq_({5, 6}, mh.predict(5))
        eq_({3}, mh.predict(3))
