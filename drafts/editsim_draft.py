import numpy as np
def editsim(mat1, mat2):
    return _simpleeditsim(mat1, mat2)
def _simpleeditsim(mat1, mat2):
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    dp_table = np.zeros((nrow+1, ncol+1))
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = np.dot(mat1[:, col1], mat2[:, col2])
            dp_table[col1 + 1, col2 + 1] = np.max([
                dp_table[col1, col2 + 1],
                dp_table[col1 + 1, col2],
                dp_table[col1, col2] + match])
    return dp_table[-1, -1]

def _simpleeditsim_withbp(mat1, mat2):
    nrow = mat1.shape[1]
    ncol = mat2.shape[1]
    dp_table = np.zeros((nrow+1, ncol+1))
    bp_table = np.ones_like(dp_table) * (-1)
    for col1 in range(nrow):
        for col2 in range(ncol):
            match = np.dot(mat1[:, col1], mat2[:, col2])
            choices = [
                dp_table[col1, col2 + 1],
                dp_table[col1 + 1, col2],
                dp_table[col1, col2] + match
            ]
            bp_table[col1 + 1, col2 + 1] = np.argmax(choices)
            dp_table[col1 + 1, col2 + 1] = np.max(choices)
            
    return dp_table, bp_table

def _simpleeditsim_tracebp(bp, mat1, mat2):
    nrow = bp.shape[0]
    ncol = bp.shape[1]
    row = nrow - 1
    col = ncol - 1
    # The first column is inserted just to avoid initialization error that may occur on concatination.
    alignment1 = np.zeros((mat1.shape[0], 1))
    alignment2 = np.zeros((mat1.shape[0], 1))
    zerovec = np.zeros(mat1.shape[0]) # which is corresponding to the null character.
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

def profile():
    nrow = 100
    mat1 = np.random.randint(0, 10, size = nrow ** 2).reshape(nrow, nrow)
    mat2 = np.random.randint(0, 10, size = nrow ** 2).reshape(nrow, nrow)
    for i in range(10):
        a = _simpleeditsim(mat1, mat2)
        print(a)