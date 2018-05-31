import numpy as np
from editsim_draft import *
def profile():
    nrow = 10
    mat1 = np.random.randint(0, 10, size = nrow ** 2).reshape(nrow, nrow)
    mat2 = np.random.randint(0, 10, size = nrow ** 2).reshape(nrow, nrow)
    for i in range(10):
        a = editsim(mat1, mat2)