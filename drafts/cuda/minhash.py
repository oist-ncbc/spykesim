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
    elif mode == "gpu":
        return _generate_signature_matrix_gpu(
            minhash.numhash, data)
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


def _generate_signature_matrix_gpu(numhash, data):
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda import gpuarray
    mod = SourceModule("""
    #include<stdio.h>
    #define A 63689
    #define B 378551
    typedef unsigned int uint;
    __device__ uint hash_fn(uint key, uint seed){
        return (A * B * (key + seed + 1)) & 0x7FFFFFFF;
    }

    __global__ void generate_signature_matrix_gpu(
        uint* signature_matrix, int numhash,
        uint* data, int nrow, int ncol){
        // c.f., https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
         for (int col = blockIdx.x * blockDim.x + threadIdx.x; 
             col < ncol; 
             col += blockDim.x * gridDim.x){
        // for (int col = 0; col < ncol; col++){
            // MinHash Start
            uint hash_ = UINT_MAX;
            uint minhash_word = 0;
            uint current_min = hash_;
            for (int idx = 0; idx < numhash; idx++){
                for (int row = 0; row < nrow; row++){
                    // calcurate minhash
                    int pos_data = row * nrow + col;
                    if (data[pos_data] != 0){
                        hash_ = hash_fn(data[pos_data], idx);
                        if (hash_ < current_min){
                            minhash_word = data[pos_data];
                            current_min = hash_;
                        }
                    }
                    if (hash_ == UINT_MAX) minhash_word = hash_fn(3511 * col, idx);
                    // MinHash end
                    signature_matrix[idx * numhash + col] = minhash_word;
                }
            }
        }
    }
    """)
    sm = np.zeros((numhash, data.shape[1]), dtype=np.uint32)
    sm_gpu = gpuarray.to_gpu(sm)
    data_gpu = gpuarray.to_gpu(data.astype(np.uint32))
    block = (1024, 1, 1)
    grid = (2, 1, 1)
    cuda_kernel = mod.get_function("generate_signature_matrix_gpu")
    cuda_kernel(sm_gpu,
                np.int32(numhash),
                data_gpu,
                np.int32(data.shape[0]),
                np.int32(data.shape[1]),
                block=block, grid=grid)
    return sm_gpu.get()


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
