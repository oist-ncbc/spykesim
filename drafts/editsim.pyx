cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport exp
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
    cdef DBL_C dp_max1 = local_exp_editsim(mat1, mat2)
    cdef DBL_C dp_max2 = local_exp_editsim(mat1, mat2[:, ::-1])
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
    dp_max1, bp1 = local_exp_editsim(mat1, mat2)
    dp_max2, bp2 = local_exp_editsim(mat1, mat2[:, ::-1])
    if dp_max1 >= dp_max2:
        return dp_max1, bp1, False
    else:
        return dp_max2, bp2, True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function        
def csimleeditsim_align(INT_C[:, :] bp, DBL_C[:, :] mat1, DBL_C[:, :] mat2_, flip=False):
    cdef INT_C nrow, ncol, nneuron
    cdef INT_C col1, col2, row, col
    if flip:
        cdef DBL_C [:, :] mat2 = mat2_[:, ::-1]
    else:
        cdef DBL_C [:, :] mat2 = mat2_
    else:
    cdef INT_C nrow = bp.shape[0]
    cdef INT_C ncol = bp.shape[1]
    cdef INT_C row = nrow - 1
    cdef INT_C col = ncol - 1
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    cdef np.ndarray[DBL_C, ndim=2] alignment1 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] alignment2 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] zerovec = np.zeros(mat1.shape[0], dtype = DBL) # which is corresponding to the null character.
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 2:
            alignment1 = np.c_[mat1[:, row - 1] * mat2[:, col - 1], alignment1]
            alignment2 = np.c_[mat1[:, row - 1] * mat2[:, col - 1], alignment2]
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

def clocal_exp_editsim_withflip(DBL_C [:, :] mat1, DBL_C [:, :] mat2, a = 0.01):
    cdef DBL_C dp_max1
    cdef INT_C dp_max_x1, dp_max_y1
    cdef DBL_C dp_max2
    cdef INT_C dp_max_x2, dp_max_y2
    dp_max1, dp_max_x1, dp_max_y1 = local_exp_editsim(mat1, mat2, a)
    dp_max2, dp_max_x2, dp_max_y2 = local_exp_editsim(mat1, mat2[:, ::-1], a)
    if dp_max1 >= dp_max2:
        return dp_max1, dp_max_x1, dp_max_y1, False
    else:
        return dp_max2, dp_max_x2, dp_max_y2, True

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function        
def local_exp_editsim_withbp(DBL_C[:, :] mat1, DBL_C[:, :] mat2, DBL_C a = 0.01):
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

def clocal_exp_editsim_withbp_withflip(DBL_C [:, :] mat1, DBL_C [:, :] mat2, a = 0.01):
    cdef DBL_C dp_max1
    cdef INT_C dp_max_x1, dp_max_y1
    cdef DBL_C dp_max2
    cdef INT_C dp_max_x2, dp_max_y2
    cdef np.ndarray[DBL_C, ndim=2] bp1, bp2
    dp_max1, dp_max_x1, dp_max_y1, bp1 = local_exp_editsim(mat1, mat2, a)
    dp_max2, dp_max_x2, dp_max_y2, bp2 = local_exp_editsim(mat1, mat2[:, ::-1], a)
    if dp_max1 >= dp_max2:
        return dp_max1, dp_max_x1, dp_max_y1, bp1, False
    else:
        return dp_max2, dp_max_x2, dp_max_y2, bp2, True

def clocal_exp_editsim_align(INT_C[:, :] bp, DBL_C[:, :] mat1, DBL_C[:, :] mat2_, flip=False):
    cdef INT_C roww, ncol, nneuron
    cdef INT_C col1, col2, row, col
    if flip:
        cdef DBL_C [:, :] mat2 = mat2_[:, ::-1]
    else:
        cdef DBL_C [:, :] mat2 = mat2_
    else:
    row = dp_max_x
    col = dp_max_y
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    cdef np.ndarray[DBL_C, ndim=2] alignment1 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] alignment2 = np.zeros((mat1.shape[0], 1), dtype = DBL)
    cdef np.ndarray[DBL_C, ndim=2] zerovec = np.zeros(mat1.shape[0], dtype = DBL) # which is corresponding to the null character.
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 3:
            alignment1 = np.c_[mat1[:, row - 1] * mat2[:, col - 1], alignment1]
            alignment2 = np.c_[mat1[:, row - 1] * mat2[:, col - 1], alignment2]
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

def eval_shrinkage(INT_C [:, :] bp, INT_C dp_max_x, INT_C dp_max_y, bint flip = False):
    cdef INT_C row = dp_max_x
    cdef INT_C col = dp_max_y
    while True:
        if bp[row, col] == -1:
            # Eather of the strings tracing terminated
            break
        elif bp[row, col] == 3:
            row -= 1
            col -= 1
        elif bp[row, col] == 2:
            col -= 1
        elif bp[row, col] == 1:
            row -= 1
        elif bp[row, col] == 0:
            break
    if flip:
        return -(dp_max_x - row) / (dp_max_y - col)
    else:
        return (dp_max_x - row) / (dp_max_y - col)
