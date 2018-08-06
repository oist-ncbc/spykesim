import numpy as np


class MinHash(object):
    def __init__(self, numband, bandwidth):
        self.numhash = numband * bandwidth
        self.numband = numband
        self.bandwidth = bandwidth


def hash_fn(key, seed=0):
    a = 63689
    b = 378551
    return (a * b * (key + seed + 1)) & 0x7FFFFFFF


def minhash(words, seed=0):
    current_min = np.inf
    minhash_word = None
    for word in words:
        hash_ = hash_fn(word, seed)
        if hash_ < current_min:
            minhash_word = word
            current_min = hash_
    return minhash_word


def generate_signature_matrix(minhash, data, mode="cpu"):
    if mode == "cpu":
        return _generate_signature_matrix_cpu(
            minhash.numhash, minhash.numband, minhash.bandwidth, data)
    elif mode == "multiprocessing":
        return _generate_signature_matrix_cpu(
            minhash.numhash, minhash.numband, minhash.bandwidth, data)
    else:
        raise RuntimeError('Option must be eather cpu or gpu')


def _generate_signature_matrix_cpu(numhash, numband, bandwidth, data):
    signature_matrix = np.zeros((numhash, data.shape[1]), dtype=np.uint32)
    for row in range(numhash):
        for col in range(data.shape[1]):
            idsets = np.where(data[:, col] >= 1)[0]
            if len(idsets) > 0:
                signature_matrix[row, col] = minhash(idsets, seed=row)
            else:
                signature_matrix[row, col] = hash_fn(3511 * col, seed=row)
    return signature_matrix


def main():
    # Test with small data
    mh = MinHash(20, 5)
    size = int(1e+02)
    data = np.random.randint(0, 1, size=size**2).reshape(size, size)

    print("Signature Matrix calculation on CPU starts")
    sm_cpu = generate_signature_matrix(mh, data, mode="cpu")
    print("Signature Matrix calculated on CPU")
    print(sm_cpu)

    print("Signature Matrix calculation on GPU starts")
    sm_gpu = generate_signature_matrix(mh, data, mode="gpu")
    print("Signature Matrix calculated on GPU")
    print(sm_gpu)

    # Performance evaluation with larger data
    import time
    mh = MinHash(20, 5)
    size = int(1e+03)
    data = np.random.randint(0, 1, size=size**2).reshape(size, size)

    print("Signature Matrix calculation on CPU starts")
    start_time = time.time()
    sm_cpu = generate_signature_matrix(mh, data, mode="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Signature Matrix calculated on CPU")
    print(sm_cpu)

    print("Signature Matrix calculation on GPU starts")
    start_time = time.time()
    sm_gpu = generate_signature_matrix(mh, data, mode="gpu")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Signature Matrix calculated on GPU")
    print(sm_gpu)


if __name__ == "__main__":
    # execute only if run as a script
    main()
