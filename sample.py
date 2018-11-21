import numpy as np
from PIL import Image
from interp import interpolation
from grad_upd import TransformDeriv 
import matplotlib.pyplot as plt
import matplotlib.colors as colr

__debug = True
# x     : a vector of xs; homogenious coordinates; dim(3xN)
# T     : transform matrix from P^3 -> P^3
# vsize : v.shape
def get_transformed( x, T, vsize) :
  L = np.dot(T, x)
  M = np.array(map(lambda x : (x[0] < vsize[0] and x[0] >= 0 and x[1] < vsize[1] and x[1] >= 0) , L)) 
  return ( x[M], L[M])

# image paths u, v; n = iterations; t = threshold of points (50); alp = learning rate
def get_transform( u, v, n, t, alp) :
  try:
    u1 = np.asarray( Image.open(u), dtype=np.float64)
    v1 = np.asarray( Image.open(v), dtype=np.float64)
  except IOError:
    return -1
  
  P1 = np.ones([3, 4*t])
  P2 = np.ones([3, 4*t])
  L1 = 0
  L2 = 0
  T = np.identity(3)
  gradV = np.gradient(v)
  gradV = (interpolate(gradV[0]), interpolate(gradV[1]))
  vintr = interpolate(v)
  for i in range(n) :
    # la
    z = 0
    while z < t :
      x = np.random.randint(u.size[1], size=[1, 4*t])
      y = np.random.randint(u.size[0], size=[1, 4*t])
      P1[1,:] = x
      P1[2,:] = y
      P1, L2 = get_transformed( P1, T, v.shape)
      z = P1.shape[1] 
    # lb
    z = 0
    while z < t :
      x = np.random.randint(u.size[1], size=[1, 4*t])
      y = np.random.randint(u.size[0], size=[1, 4*t])
      P2[1,:] = x
      P2[2,:] = y
      P2, L2 = get_transformed( P2, T, v.shape)
      z = P2.shape[1] 
    
    if __debug :
      T1 = np.linalg.inv(T)
      x = T1[:,0]@np.tile(range(v.shape[1]), v.shape[0], 1).flatten() 
      y = T1[:,1]@np.tile(np.array([range(v.shape[0])]).T, 1, v.shape[1]).flatten()
      q = T1[:,2]@np.ones([1,v.shape[0]*v.shape[1]])
      x = x + y + q
      v1 = vintr(x[1,:], x[0,:]).reshape(v.shape)
      plt.imshow(v1, norm=colr.Normalize().autoscale(v1));
      plt.pause()

# def TransformDeriv(T,u,la,lb,CovInvU,CovInvV,Vinterp,gradVx,gradVy):
    
    Tn = TransformDer(T, u, P1, P2, 0.5, 0.5, vintr, gradV[1], gradV[0])  
    T[0, :] = T[0, :] + (alp* Tn[0,:])
    T[1, :] = T[1, :] + (alp* Tn[1,:])
  return T 
