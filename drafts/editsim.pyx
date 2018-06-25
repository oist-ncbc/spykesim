cimport cython
import numpy as np
cimport numpy as np
DBL = np.double
ctypedef np.double_t DBL_C


cdef inline float float_min2(float a, float b):
    return a if a <= b else b
cdef inline float float_min(float a, float b, float c):
    cdef float d = float_min2(a, b)
    cdef float e = float_min2(b, c)
    return float_min2(d, e)
cdef inline float float_max2(float a, float b):
    return a if a >= b else b
cdef inline float float_max(float a, float b, float c):
    cdef float d = float_max2(a, b)
    cdef float e = float_max2(b, c)
    return float_max2(d, e)
cdef inline int int_max(int a, int b):
    return a if a >= b else b
cdef inline int int_min(int a, int b):
    return a if a <= b else b
cdef struct Pair:
    float value
    int idx
cdef inline Pair indmax(float a, float b, float c):
    cdef Pair pair
    if a >= b:
        if a >= c:
            pair.value = a
            pair.idx = 0
            return pair
        else:
            # b <= a <= c
            pair.value = c
            pair.idx = 2
            return pair
    else:
        # a <= b
        if b >= c:
            # a < b and b > c
            pair.value = b
            pair.idx = 1
            return pair
        else:
            # a < b and b < c
            pair.value = c
            pair.idx = 2
            return pair


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(True)  # turn off negative index wrapping for entire function
def csimpleeditsim(DBL_C [:, :] mat1, DBL_C [:, :] mat2):
    cdef int nrow, ncol, nneuron
    cdef int col1, col2, row
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    nneuron = mat1.shape[0]
    cdef np.ndarray[DBL_C, ndim=2] dp_table = np.zeros((nrow+1, ncol+1), dtype = DBL)
    cdef DBL_C match
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = 0
            for row in range(nneuron):
                match += mat1[row, col1] * mat2[row, col2]
            dp_table[col1 + 1, col2 + 1] = float_max(
                dp_table[col1, col2 + 1],
                dp_table[col1 + 1, col2],
                dp_table[col1, col2] + match)
    return dp_table[-1, -1]