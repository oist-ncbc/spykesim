cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp
from libc.math cimport log
import multiprocessing
import ctypes
from functools import partial
import os
from .parallel import parallel_process
from .minhash import MinHash, generate_signature_matrix_cpu_multi, generate_bucket_list_single, find_similar, generate_signature_matrix_cpu_single
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, coo_matrix
from pathlib import Path
import h5py
import datetime
from logging import StreamHandler, Formatter, INFO, getLogger
from hdbscan import HDBSCAN
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd
from scipy.ndimage.measurements import center_of_mass

def init_logger():
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(Formatter("[%(asctime)s] [%(threadName)s] %(message)s"))
    logger = getLogger()
    if (logger.hasHandlers()):
        logger.handlers.clear()    
    logger.addHandler(handler)
    logger.setLevel(INFO)

class FromBinMat(object):
    """Compute extended-edit similarity values of segments of a binned-multineuronal activity data

    Find more in the References.

    Parameters
    ------------
    sim_type: {"exp", "linear", "simple"}
        Form of the penalization. Currently, only "exp" is implemented.

    alpha: float, default: 0.1
        Strength of gap penalty used in "exp" and "linear".

    reverse: bool
        Whether consider reverse patterns(True) or not(False)

    Attributes
    ------------
    binarray_csc: csc sparse matrix
        The binned-multi neuronal activity matrix. This will be set via ``gensimmat``. Row indicates neuron ID and column indicates time bins.

    window: int 
        The size of time window. This will be set when ``gensimmat`` performed.

    simmat: array of shap = [number of segments, number of segments] 
        The similarity matrix when ``gensimmat`` performed.


    Examples
    ------------
    >>> es = editsim.FromBinMat(alpha=0.01)
    >>> sim, _, _, _ = es._sim(mat1, mat2)
    >>> print(f"The edit similarity between mat1 and mat2 is {sim}")

    >>> window = 100
    >>> es.gensimmat(
    ...     binmat, window=window, slide=window, numband=4, bandwidth=5, minhash=False
    ... )

    References
    -------------
    bioArxiv: https://www.biorxiv.org/content/early/2017/10/30/202655


    """
    def __init__(self, sim_type="exp", alpha=0.1, reverse=True):
        # すべてのAttributesをデフォルトNoneで宣言しておいたほうが、save, load functionで気を使わなくて良くなっていいかもしれない。
        self.sim_type = sim_type
        self.alpha = alpha
        self.reverse = reverse
        if sim_type=="exp" and reverse:
            _sim = partial(
                clocal_exp_editsim_withflip,
                a=alpha
            )
            _sim.__name__ = "editsim_expgap"
            self._sim = _sim
            _sim_bp = partial(
                clocal_exp_editsim_withbp_withflip,
                a=alpha
            )
            _sim_bp.__name__ = "editsim_expgap_withbp"
            self._sim_bp = _sim_bp
        elif sim_type=="exp" and not reverse:
            raise NotImplementedError()
        elif sim_type=="linear" and reverse:
            raise NotImplementedError()
        elif sim_type=="linear" and not reverse:
            raise NotImplementedError()
        else:
            raise AttributeError("The option is not supported.")

    def sim(self, csc_mat1, csc_mat2, with_bp=False):
        if with_bp:
            return self._sim_bp(csc_mat1, csc_mat2)
        else:
            return self._sim(csc_mat1, csc_mat2)

    def gensimmat(self, binarray_csc, window, slide,
               minhash=True, numband=5, bandwidth=10, njobs=os.cpu_count()):
        """Compute the edit similarity matrix. 

        Find more in the References.

        Parameters
        ------------
        window: int 
            The size of time window. 

        slide: int
            The size of sliding for the time window.

        minhash: bool
            If True, the minhash algorithm will be performed to 

        Attributes
        ------------
        binarray_csc: csc sparse matrix
            The binned-multi neuronal activity matrix.

        simmat: array of shap = [number of segments, number of segments] 
            The similarity matrix when ``gensimmat`` performed.

        reduce_rate: float
            This shows how much percent of pairs calculated. It will be set only when minhash=True


        Examples
        ------------
        >>> window = 100
        >>> es.gensimmat(
        ...     binmat, window=window, slide=window, numband=4, bandwidth=5, minhash=False
        ... )

        References
        -------------
        bioArxiv: https://www.biorxiv.org/content/early/2017/10/30/202655
        """

        self.binarray_csc = binarray_csc
        self.window = window
        self.slide = slide
        self.minhash = minhash
        init_logger()
        getLogger().info("Execution of a function gensimmat starts")
        if minhash:
            self.numband = numband
            self.bandwidth = bandwidth
            times = None
            numhash = numband * bandwidth
            self.simmat, self.times, self.reduce_rate =  _eval_simmat_minhash(
                self._sim, numhash, numband, bandwidth, binarray_csc, window, slide, njobs)
            getLogger().info(f"Reduce Rate: {self.reduce_rate}")
        else:
            nneuron, duration = binarray_csc.shape
            times = np.arange(0, duration-window, slide)
            self.simmat, self.times = _eval_simmat(
                    self._sim, times, binarray_csc, window, slide, minhash)

    def clustering(self, min_cluster_size=5):
        """
        Perform HDBSCAN clustering algorithm on the similarity matrix calculated by `gensimmat`

        """
        self.clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        self.cluster_labels = self.clusterer.fit_predict(self.simmat)

    def gen_profile(self, th_=5, sigma=5):
        uidxs = []
        mats_list = []
        self.th_ = th_
        self.sigma = sigma
        for uidx in np.unique(self.cluster_labels):
            # uidx == -1 is just noise 
            if uidx == -1:
                continue
            indices = np.where(self.cluster_labels == uidx)[0]
            mats = []
            mats_org = []
            for idx in indices:
                mat = self.binarray_csc[:, self.window*idx:(self.window*idx+self.window)].toarray()
                if mat.sum() >= th_:
                    mats_org.append(
                        mat.copy()
                    )                    
                    for row in range(mat.shape[0]):
                        mat[row, :] = gaussian_filter1d(input=mat[row, :], sigma=sigma)
                    mats.append(
                        mat.copy()
                    )
            if len(mats) >= 5:
                mats = np.array(mats)
                uidxs.append(uidx)
                mats_list.append(mats.copy())
        self.profiles = dict()
        for uidx, mats in zip(uidxs, mats_list):
            profile = regularize_profile(barton_sternberg(mats, self._sim_bp, 2*len(mats)))
            if profile.sum() >= th_:
                self.profiles[uidx] = regularize_profile(barton_sternberg(mats, self._sim_bp, 2*len(mats)))
        return self           

    def detect_sequences(self, cluster_id):
        raise NotImplementedError()

    def save(self, path="."):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        d = datetime.datetime.today()
        simmat_file = f"simmat_{self.window}_{self.slide}_{self.alpha}_{self.sim_type}.hdf5"
        with h5py.File(path / simmat_file, "w") as wf:
            wf.create_dataset("window", data=self.window)
            wf.create_dataset("slide", data=self.slide)
            wf.create_dataset("alpha", data=self.alpha)
            wf.create_dataset("minhash", data=self.minhash)
            if self.minhash:
                wf.create_dataset("bandwidth", data=self.bandwidth)
                wf.create_dataset("numband", data=self.numband)
            wf.create_dataset("row", data=self.simmat.row)
            wf.create_dataset("col", data=self.simmat.col)
            wf.create_dataset("data", data=self.simmat.data)
            wf.create_dataset("times", data=self.times)
    
    def load(self, hdf5_file):
        with h5py.File(hdf5_file, 'r') as rf:
            self.window = rf["window"].value
            self.slide = rf["slide"].value
            self.alpha = rf["alpha"].value
            self.minhash = rf["minhash"].value
            if self.minhash:
                self.bandwidth = rf["bandwidth"].value
                self.numband = rf["numband"].value
            row = rf["row"].value
            col = rf["col"].value
            data = rf["data"].value
            self.simmat = coo_matrix((data, (row, col)))
            self.times = rf["times"].value
            

        
DBL = np.double
ctypedef np.double_t DBL_C
INT = np.int
ctypedef np.int_t INT_C

cdef inline DBL_C max2(DBL_C a, DBL_C b):
    return a if a >= b else b
cdef inline DBL_C max3(DBL_C a, DBL_C b, DBL_C c):
    return max2(max2(a, b), c)
cdef inline DBL_C max4(DBL_C a, DBL_C b, DBL_C c, DBL_C d):
    return max2(max3(a, b, c), d)

cdef struct Pair:
    int idx
    DBL_C value
cdef inline Pair assign(int idx, DBL_C value):
    cdef Pair pair
    pair.idx = idx
    pair.value = value
    return pair
cdef inline Pair pairmax2_(Pair pair1, Pair pair2):
    return pair1 if pair1.value >= pair2.value else pair2
cdef inline Pair pairmax3_(Pair pair1, Pair pair2, Pair pair3):
    return pairmax2_(pairmax2_(pair1, pair2), pair3)
cdef inline Pair pairmax4_(Pair pair1, Pair pair2, Pair pair3, Pair pair4):
    return pairmax2_(pairmax3_(pair1, pair2, pair3), pair4)
cdef inline Pair pairmax2(DBL_C a, DBL_C b):
    return pairmax2_(assign(0, a), assign(1, b))
cdef inline Pair pairmax3(DBL_C a, DBL_C b, DBL_C c):
    return pairmax3_(assign(0, a), assign(1, b), assign(2, c))
cdef inline Pair pairmax4(DBL_C a, DBL_C b, DBL_C c, DBL_C d):
    return pairmax4_(assign(0, a), assign(1, b), assign(2, c), assign(3, d))

cdef inline DBL_C cexp(DBL_C a, INT_C x):
    return exp(a*x)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function
def csimpleeditsim(DBL_C [:, :] mat1, DBL_C [:, :] mat2):
    cdef int nrow, ncol, nneuron
    cdef int col1, col2, row
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    nneuron = mat1.shape[0]
    cdef np.ndarray[DBL_C, ndim=2] dp = np.zeros((nrow+1, ncol+1), dtype = DBL)
    cdef DBL_C match
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = 0
            for row in range(nneuron):
                match += mat1[row, col1] * mat2[row, col2]
            dp[col1 + 1, col2 + 1] = max3(
                dp[col1, col2 + 1],
                dp[col1 + 1, col2],
                dp[col1, col2] + match)
    return dp[-1, -1]

def csimpleeditsim_withflip(DBL_C [:, :] mat1, DBL_C [:, :] mat2):
    cdef DBL_C dp_max1 = csimpleeditsim(mat1, mat2)
    cdef DBL_C dp_max2 = csimpleeditsim(mat1, mat2[:, ::-1])
    if dp_max1 >= dp_max2:
        return dp_max1, False
    else:
        return dp_max2, True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function        
def csimpleeditsim_withbp(DBL_C [:, :] mat1, DBL_C [:, :] mat2):
    cdef int nrow, ncol, nneuron
    cdef int col1, col2, row    
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    nneuron = mat1.shape[0]
    cdef np.ndarray[DBL_C, ndim=2] dp = np.zeros((nrow+1, ncol+1), dtype = DBL)
    cdef np.ndarray[INT_C, ndim=2] bp = np.ones_like(dp, dtype = INT) * (-1)
    cdef DBL_C match
    cdef Pair pair
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = 0
            for row in range(nneuron):
                match += mat1[row, col1] * mat2[row, col2]            
            pair = pairmax3(
                dp[col1, col2 + 1],
                dp[col1 + 1, col2],
                dp[col1, col2] + match
            )
            bp[col1 + 1, col2 + 1] = pair.idx
            dp[col1 + 1, col2 + 1] = pair.value
    return dp[-1, -1], bp
def csimpleeditsim_withbp_withflip(DBL_C [:, :] mat1, DBL_C [:, :] mat2):
    cdef DBL_C dp_max1, dp_max2
    cdef np.ndarray[DBL_C, ndim=2] bp1, bp2
    dp_max1, bp1 = csimpleeditsim_withbp(mat1, mat2)
    dp_max2, bp2 = csimpleeditsim_withbp(mat1, mat2[:, ::-1])
    if dp_max1 >= dp_max2:
        return dp_max1, bp1, False
    else:
        return dp_max2, bp2, True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function        
def csimleeditsim_align(INT_C[:, :] bp, DBL_C[:, :] mat1, DBL_C[:, :] mat2_, flip=False):
    cdef INT_C nrow, ncol, nneuron
    cdef INT_C col1, col2, row, col, i
    cdef DBL_C [:, :] mat2
    if flip:
        mat2 = mat2_[:, ::-1]
    else:
        mat2 = mat2_
    nneuron = mat1.shape[0]
    nrow = bp.shape[0]
    ncol = bp.shape[1]
    row = nrow - 1
    col = ncol - 1
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    cdef np.ndarray[DBL_C, ndim=2] alignment1 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] alignment2 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] zerovec = np.zeros(mat1.shape[0], dtype = DBL) # which is corresponding to the null character.
    cdef np.ndarray[DBL_C, ndim=1] match = np.zeros(mat1.shape[0], dtype = DBL)
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 2:
            for i in range(nneuron):
                match[i] = mat1[i, row] * mat2[i, col]            
            alignment1 = np.c_[match, alignment1]
            alignment2 = np.c_[match, alignment2]
            row -= 1
            col -= 1
        elif bp[row, col] == 1:
            alignment1 = np.c_[zerovec, alignment1]
            alignment2 = np.c_[mat2[:, col - 1], alignment2]
            col -= 1
        elif bp[row, col] == 0:
            alignment1 = np.c_[mat1[:, row - 1], alignment1]
            alignment2 = np.c_[zerovec, alignment2]
            row -= 1
    return alignment1[:, :-1], alignment2[:, :-1]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function        
def clocal_exp_editsim(DBL_C[:, :] mat1, DBL_C[:, :] mat2, DBL_C a = 0.01):
    cdef int nrow, ncol, nneuron
    cdef int col1, col2, row      
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    nneuron = mat1.shape[0]
    cdef np.ndarray[DBL_C, ndim=2] dp = np.zeros((nrow+1, ncol+1), dtype = DBL)
    cdef DBL_C dp_max = 0
    cdef INT_C dp_max_x = -1
    cdef INT_C dp_max_y = -1
    cdef np.ndarray[INT_C, ndim=2] down = np.zeros((nrow+1, ncol+1), dtype = INT)
    cdef np.ndarray[INT_C, ndim=2] right = np.zeros((nrow+1, ncol+1), dtype = INT)
    cdef DBL_C match
    cdef Pair dirpair
    cdef DBL_C down_score,right_score
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = 0
            for row in range(nneuron):
                match += mat1[row, col1] * mat2[row, col2]            
            match = -10 if match == 0 else match
            # For Down
            # Comparing options: newly extend a gap or extend the exsinting gap
            dirpair = pairmax2(
                dp[col1, col2+1] - cexp(a, 1) + 1,
                dp[col1-down[col1, col2+1], col2+1] - cexp(a, down[col1, col2+1] + 1) + 1
            )
            down[col1+1, col2+1] = 1 if dirpair.idx == 0 else down[col1, col2+1] + 1
            down_score = dp[col1-down[col1+1, col2+1]+1, col2+1] - cexp(a, down[col1+1, col2+1]) + 1
            # For Rightp
            # Comparing options: newly extend a gap or extend the exsinting gap
            dirpair = pairmax2(
                dp[col1+1, col2] - cexp(a, 1) + 1,
                dp[col1+1, col2-right[col1+1, col2]] - cexp(a, right[col1+1, col2] + 1) + 1
            )
            right[col1+1, col2+1] = 1 if dirpair.idx == 0 else right[col1+1, col2] + 1
            right_score = dp[col1+1, col2-right[col1+1, col2+1]+1] - cexp(a, right[col1+1, col2+1]) + 1
            # Update dp
            dp[col1+1, col2+1] = max4(
                0,
                down_score,
                right_score,
                dp[col1, col2] + match
            )
            if dp[col1+1, col2+1] > dp_max:
                dp_max = dp[col1+1, col2+1]
                dp_max_x = col1 + 1
                dp_max_y = col2 + 1
    return dp_max, dp_max_x, dp_max_y        

def clocal_exp_editsim_withflip(DBL_C [:, :] mat1, DBL_C [:, :] mat2, DBL_C a = 0.01):
    cdef DBL_C dp_max1
    cdef INT_C dp_max_x1, dp_max_y1
    cdef DBL_C dp_max2
    cdef INT_C dp_max_x2, dp_max_y2
    dp_max1, dp_max_x1, dp_max_y1 = clocal_exp_editsim(mat1, mat2, a)
    dp_max2, dp_max_x2, dp_max_y2 = clocal_exp_editsim(mat1, mat2[:, ::-1], a)
    if dp_max1 >= dp_max2:
        return dp_max1, dp_max_x1, dp_max_y1, False
    else:
        return dp_max2, dp_max_x2, dp_max_y2, True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function        
cpdef clocal_exp_editsim_withbp(DBL_C[:, :] mat1, DBL_C[:, :] mat2, DBL_C a = 0.01):
    cdef int nrow, ncol, nneuron
    cdef int col1, col2, row      
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    nneuron = mat1.shape[0]
    cdef np.ndarray[DBL_C, ndim=2] dp = np.zeros((nrow+1, ncol+1), dtype = DBL)
    cdef np.ndarray[INT_C, ndim=2] bp = np.zeros((nrow+1, ncol+1), dtype = INT)
    cdef DBL_C dp_max = 0
    cdef INT_C dp_max_x = -1
    cdef INT_C dp_max_y = -1
    cdef np.ndarray[INT_C, ndim=2] down = np.zeros((nrow+1, ncol+1), dtype = INT)
    cdef np.ndarray[INT_C, ndim=2] right = np.zeros((nrow+1, ncol+1), dtype = INT)
    cdef DBL_C match
    cdef Pair dirpair
    cdef DBL_C down_score,right_score
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = 0
            for row in range(nneuron):
                match += mat1[row, col1] * mat2[row, col2]            
            match = -10 if match == 0 else match
            # For Down
            # Comparing options: newly extend a gap or extend the exsinting gap
            dirpair = pairmax2(
                dp[col1, col2+1] - cexp(a, 1) + 1,
                dp[col1-down[col1, col2+1], col2+1] - cexp(a, down[col1, col2+1] + 1) + 1
            )
            down[col1+1, col2+1] = 1 if dirpair.idx == 0 else down[col1, col2+1] + 1
            down_score = dp[col1-down[col1+1, col2+1]+1, col2+1] - cexp(a, down[col1+1, col2+1]) + 1
            # For Rightp
            # Comparing options: newly extend a gap or extend the exsinting gap
            dirpair = pairmax2(
                dp[col1+1, col2] - cexp(a, 1) + 1,
                dp[col1+1, col2-right[col1+1, col2]] - cexp(a, right[col1+1, col2] + 1) + 1
            )
            right[col1+1, col2+1] = 1 if dirpair.idx == 0 else right[col1+1, col2] + 1
            right_score = dp[col1+1, col2-right[col1+1, col2+1]+1] - cexp(a, right[col1+1, col2+1]) + 1
            # Update dp
            pair = pairmax4(
                0,
                down_score,
                right_score,
                dp[col1, col2] + match
            )
            bp[col1 + 1, col2 + 1] = pair.idx
            dp[col1 + 1, col2 + 1] = pair.value
            if dp[col1+1, col2+1] > dp_max:
                dp_max = dp[col1+1, col2+1]
                dp_max_x = col1 + 1
                dp_max_y = col2 + 1
    return dp_max, dp_max_x, dp_max_y, bp

def clocal_exp_editsim_withbp_withflip(DBL_C[:, :] mat1, DBL_C[:, :] mat2, DBL_C a = 0.01):
    cdef DBL_C dp_max1
    cdef INT_C dp_max_x1, dp_max_y1
    cdef DBL_C dp_max2
    cdef INT_C dp_max_x2, dp_max_y2
    cdef np.ndarray[INT_C, ndim=2] bp1, bp2
    dp_max1, dp_max_x1, dp_max_y1, bp1 = clocal_exp_editsim_withbp(mat1, mat2, a)
    dp_max2, dp_max_x2, dp_max_y2, bp2 = clocal_exp_editsim_withbp(mat1, mat2[:, ::-1], a)
    if dp_max1 >= dp_max2:
        return dp_max1, dp_max_x1, dp_max_y1, bp1, False
    else:
        return dp_max2, dp_max_x2, dp_max_y2, bp2, True

def clocal_editsim_align(INT_C[:, :] bp, INT_C dp_max_x, INT_C dp_max_y, DBL_C[:, :] mat1, DBL_C[:, :] mat2_, flip=False):
    cdef INT_C roww, ncol, nneuron
    cdef INT_C col1, col2, row, col, i
    cdef DBL_C [:, :] mat2
    if flip:
        mat2 = mat2_[:, ::-1]
    else:
        mat2 = mat2_
    nneuron = mat1.shape[0]
    row = dp_max_x
    col = dp_max_y
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    cdef np.ndarray[DBL_C, ndim=2] alignment1 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] alignment2 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] zerovec = np.zeros(mat1.shape[0], dtype = DBL) # which is corresponding to the null character.
    cdef np.ndarray[DBL_C, ndim=1] match = np.zeros(mat1.shape[0], dtype = DBL)
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 3:
            for i in range(nneuron):
                match[i] = mat1[i, row] * mat2[i, col]            
            alignment1 = np.c_[match, alignment1]
            alignment2 = np.c_[match, alignment2]
            row -= 1
            col -= 1
        elif bp[row, col] == 2:
            alignment1 = np.c_[zerovec, alignment1]
            alignment2 = np.c_[mat2[:, col - 1], alignment2]
            col -= 1
        elif bp[row, col] == 1:
            alignment1 = np.c_[mat1[:, row - 1], alignment1]
            alignment2 = np.c_[zerovec, alignment2]
            row -= 1
        elif bp[row, col] == 0:
            break
        while row > 1:
            alignment1 = np.c_[mat1[:, row - 1], alignment1]
            row = row - 1
        while col > 1:
            alignment2 = np.c_[mat2[:, col - 1], alignment2]
            col = col - 1
    return alignment1[:, :-1], alignment2[:, :-1]


def clocal_editsim_align_alt(INT_C[:, :] bp, INT_C dp_max_x, INT_C dp_max_y, DBL_C[:, :] mat1, DBL_C[:, :] mat2_, flip=False):
    cdef INT_C roww, ncol, nneuron
    cdef INT_C col1, col2, row, col, i
    cdef DBL_C [:, :] mat2
    if flip:
        mat2 = mat2_[:, ::-1]
    else:
        mat2 = mat2_
    nneuron = mat1.shape[0]
    row = dp_max_x
    col = dp_max_y
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    cdef np.ndarray[DBL_C, ndim=2] alignment1 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] alignment2 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] zerovec = np.zeros(mat1.shape[0], dtype = DBL) # which is corresponding to the null character.
    cdef np.ndarray[DBL_C, ndim=1] match = np.zeros(mat1.shape[0], dtype = DBL)
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 3:
            for i in range(nneuron):
                match[i] = mat1[i, row] * mat2[i, col]            
            alignment1 = np.c_[match, alignment1]
            alignment2 = np.c_[match, alignment2]
            row -= 1
            col -= 1
        elif bp[row, col] == 2:
            alignment2 = np.c_[zerovec, alignment2]
            col -= 1
        elif bp[row, col] == 1:
            alignment1 = np.c_[zerovec, alignment1]
            row -= 1
        elif bp[row, col] == 0:
            break
        while row > 1:
            alignment1 = np.c_[zerovec, alignment1]
            row = row - 1
        while col > 1:
            alignment2 = np.c_[zerovec, alignment2]
            col = col - 1
    return alignment1[:, :-1], alignment2[:, :-1]

def _eval_simvec(_sim, idx1, t1, times, binarray_csc, window):
    simvec = np.zeros(len(times))
    m1 = binarray_csc[:, t1:(t1+window)].toarray().astype(DBL)
    for idx2, t2 in enumerate(times):
        m2 = binarray_csc[:, t2:(t2+window)].toarray().astype(DBL)
        dp_max, _, _, _ = _sim(m1, m2)
        simvec[idx2] = dp_max
    return (simvec, idx1)

def _eval_simmat(_sim, times, binarray_csc, INT_C window = 200, INT_C slidewidth = 200, bint lsh=False, njobs = -1):
    njobs = os.cpu_count() if njobs == -1 else njobs
    simmat = np.zeros((len(times), len(times)))
    worker = partial(
        _eval_simvec,
        _sim = _sim,
        times = times,
        binarray_csc = binarray_csc,
        window = window
    )
    worker.__name__ = _eval_simvec.__name__
    args = [{
        "idx1": idx1,
        "t1": t1,
        "times": times,
        "binarray_csc": binarray_csc,
        "window": window,
    } for idx1, t1 in enumerate(times)]
    results = parallel_process(args, worker, njobs, use_kwargs=True)
    for simvec, idx1 in results:
        simmat[idx1, :] = simvec
    return simmat, times

def _get_nonzero_indices(idx, indices, indptr, col, span):
    """
    get nonzero indices from indices and indptr of a csr matrix
    """
    return idx, indices[indptr[col]:indptr[col+span]]

def _get_idmat_multi(times, binarray_csc, window, njobs):
    """
    Currently single core version is used due to unsolved bug in this function.
    """
    worker = partial(
        _get_nonzero_indices,
        indices=binarray_csc.indices,
        indptr=binarray_csc.indptr,
        span=window
    )
    worker.__name__ = _get_nonzero_indices.__name__
    args = [{
        "idx": idx,
        "col": col
    } for idx, col in enumerate(times)] 
    results = parallel_process(args, worker, njobs, use_kwargs=True)
    idmat = np.empty((binarray_csc.shape[0], len(times)))
    for idx, indices in results:
        idmat[indices, idx] = 1
    return csc_matrix(idmat)
def _get_idmat(times, binarray_csc, window):
    idmat = np.zeros((binarray_csc.shape[0], len(times)))
    for idx, col in enumerate(times):
        idmat[:, idx] = binarray_csc[:, col:(col+window)].sum(axis=1).flatten()
    return csc_matrix(idmat)

def _eval_simvec_lsh(_sim, idx1, t1, len_times, indices, times, binarray_csc, window):
    simvec = np.zeros(len_times)
    m1 = binarray_csc[:, t1:(t1+window)].toarray().astype(DBL)
    for idx2, t2 in zip(indices, times):
        m2 = binarray_csc[:, t2:(t2+window)].toarray().astype(DBL)
        dp_max, _, _, _ = _sim(m1, m2)
        simvec[idx2] = dp_max
    return (simvec, idx1)

def _eval_simmat_minhash(_sim, numhash, numband, bandwidth, binarray_csc, INT_C window = 200, INT_C slidewidth = 200, njobs=12):
    times = range(0, binarray_csc.shape[1] - window, slidewidth)
    len_times = len(times)
    # idmat = _get_idmat_multi(times, binarray_csc, window, njobs)
    idmat = _get_idmat(times, binarray_csc, window)
    # sigmat = generate_signature_matrix_cpu_multi(numhash, numband, bandwidth, idmat, njobs)
    sigmat = generate_signature_matrix_cpu_single(numhash, numband, bandwidth, idmat)
    bucket_list = generate_bucket_list_single(numhash, numband, bandwidth, sigmat)
    indices_list = []
    times_list = []
    count = 0
    for idx1, t1 in enumerate(times):
        indices = find_similar(numhash, numband, bandwidth, sigmat, bucket_list, idx1)
        indices_list.append(set(indices))
        count += len(set(indices))
        times_list.append([times[idx2] for idx2 in indices])
    reduce_rate = (count / (len_times ** 2))
    worker = partial(
        _eval_simvec_lsh,
        _sim = _sim,
        binarray_csc = binarray_csc,
        len_times = len_times,
        window = window,
    )
    worker.__name__ = _eval_simvec_lsh.__name__
    args = [{
        "idx1": idx1,
        "t1": t1,
        "indices": indices,
        "times": times,
        "binarray_csc": binarray_csc,
        "window": window,
    } for (idx1, t1), indices, times in zip(
        enumerate(times),
        indices_list,
        times_list)
    ]
    results = parallel_process(args, worker, njobs, use_kwargs=True)
    simmat_lil = lil_matrix((len_times, len_times))
    for simvec, idx1 in results:
        simmat_lil[idx1, :] = simvec

    return simmat_lil.tocoo(), times, reduce_rate


def sort_profile(profile):
    coms = list()        
    for row in range(profile.shape[0]):
        com = center_of_mass(profile[row, :])[0]
        if not np.isnan(com) and not np.isinf(com):
            coms.append(int(com))
        else:
            coms.append(profile.shape[1]-1)
    coms = pd.DataFrame({"CoM" : coms})    
    coms = coms.sort_values(by="CoM")
    sorted_profile = np.zeros_like(profile)
    for srow, row in enumerate(coms.index):
        sorted_profile[srow, :] = profile[row, :]
    return(coms.index, sorted_profile)    

def regularize_profile(profile_):
    profile = profile_.copy()
    for row in range(profile.shape[0]):
        if profile[row, :].sum() > 0:
            profile[row, :] = (profile[row, :] - profile[row, :].min()) / (profile[row, :].max() - profile[row, :].min())
    return profile
def barton_sternberg(mats_, sim_bp, niter):
    # First profile generation
    mats = mats_.copy()
    simmat = np.zeros((len(mats), len(mats)))
    processed = np.zeros(len(mats), dtype=np.bool)
    #for idx1, mat1 in enumerate(mats):
    #    for idx2, mat2 in enumerate(mats[:idx1]):
    #        dp_max, dp_max_x, dp_max_y, bp, flip = sim_bp(
    #            mat1.astype(np.double), mat2.astype(np.double))
    #        simmat[idx1, idx2] = dp_max
    #i, j = np.unravel_index(simmat.argmax(), simmat.shape)
    i, j = 1, 2 # for test
    dp_max, dp_max_x, dp_max_y, bp, flip = sim_bp(
        mats[i].astype(np.double), mats[j].astype(np.double))
    al1, al2 = clocal_exp_editsim_align_alt(bp, dp_max_x, dp_max_y, mats[i], mats[j], flip)
    mats[i] = al1
    mats[j] = al2
    al = (al1 + al2) / 2
    processed[i] = True
    processed[j] = True
    simvec = np.zeros(len(mats))
    for niter_ in tqdm(range(1, niter)): 
        if processed.sum() == 0:
            processed = np.zeros(len(mats), dtype=np.bool)
        else:
            for idx, mat2 in zip(np.arange(len(mats))[~processed], mats[~processed]):
                dp_max, dp_max_x, dp_max_y, bp, flip = sim_bp(
                    al.astype(np.double), mat2.astype(np.double))
                simvec[idx] = dp_max
            j = simvec.argmax()
            mat2 = mats[j]
            dp_max, dp_max_x, dp_max_y, bp, flip = sim_bp(
                al.astype(np.double), mats[j].astype(np.double))
            al1, al2 = clocal_exp_editsim_align_alt(bp, dp_max_x, dp_max_y, al, mats[j], flip)
            al = (al1 + al2) / 2            
            mats[j] = al2
            processed[j] = True
    return mats[i]

def clocal_exp_editsim_align_alt(bp, dp_max_x, dp_max_y, mat1, mat2_, flip=False):
    if flip:
        mat2 = mat2_[:, ::-1]
    else:
        mat2 = mat2_
    nneuron = mat1.shape[0]
    row = dp_max_x
    col = dp_max_y
    row_ = 0
    col_ = 0
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    alignment1 = np.zeros_like(mat1)
    alignment2 = np.zeros_like(mat2)
    zerovec = np.zeros(mat1.shape[0]) # which is corresponding to the null character.
    match = np.zeros(mat1.shape[0])
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 3:
#             for i in range(nneuron):
#                 match[i] = mat1[i, row-1] * mat2[i, col-1]
            alignment1[:, row_] = mat1[:, row-1] + mat2[:, col-1]
            alignment2[:, col_] = mat1[:, row-1] + mat2[:, col-1]
            row -= 1
            col -= 1
            row_ += 1
            col_ += 1
        elif bp[row, col] == 2:
            col -= 1
            col_ += 1
            alignment2[:, col_] = mat2[:, col-1]
        elif bp[row, col] == 1:
            row -= 1
            row_ += 1
            alignment1[:, row_] = mat1[:, row-1]
        elif bp[row, col] == 0:
            break
    alignment1[:, :] = alignment1[:, ::-1]
    alignment2[:, :] = alignment2[:, ::-1]
    return alignment1, alignment2
        
