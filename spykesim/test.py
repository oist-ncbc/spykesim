import numpy as np

from joblib import Parallel, delayed, cpu_count

import editsim

def su(a,i):
  return a[i].sum();

def execute():
  print("test")
  dims = (100000,4);
  x = editsim.createSharedNumpyArray(dims);
  x[:] = np.random.rand(dims[0], dims[1]);
  res = Parallel(n_jobs = cpu_count())(delayed(su)(x,i) for i in range(dims[0]));
  print(res)

execute()
