import numpy as np

# x     : a vector of xs; homogenious coordinates; dim(3xN)
# T     : transform matrix from P^3 -> P^3
# vsize : v.shape
def get_transformed( x, T, vsize) :
  L = np.dot(T, x)
  M = np.array(map(lambda x : (x[0] < vsize[0] && x[0] >= 0 && x[1] < vsize[1] && x[1] >= 0) , L)) 
  return ( x[M], L[M])
