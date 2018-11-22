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
def get_filter( x, T, vsize, usize) :
  L = T@x
  B = T@[[usize[1]/2.0], [usize[0]/2.0], [0]] - [[vsize[1]/2.0], [vsize[0]/2.0], [0]]
  H = B + [[vsize[1]], [vsize[0]], [0]]
  # print(L[0:1])
  # L=L.astype(np.int)
  # print(type(L[1,1]))
  # L = np.dot(T, x)
  # print(x)
  # print(T)
  M = map(lambda x : x[0] < H[0] and x[0] >= B[0] and x[1] < H[1] and x[1] >= B[1] , L.T)
  M = np.array(list(M)).T[0]
  return ( (x.T[M]).T, (L.T[M]).T)

# image paths u, v; n = iterations; t = threshold of points (50); alp = learning rate
def get_transform( u1, v1, n, t, alp) :
  try:
    u = np.asarray( Image.open(u1), dtype=np.float64)
    u=u/np.max(u)
    v = np.asarray( Image.open(v1), dtype=np.float64)
    v=v/np.max(v)
  except IOError:
    return -1
  
  L1 = 0
  L2 = 0
  T = np.identity(3)
  gradV = np.gradient(v)
  # print(v)
  gradV = (interpolation(gradV[0]), interpolation(gradV[1]))
  vintr = interpolation(v)
  # for x in range(1,v.shape[0]):
  #   for y in range(1,v.shape[1]):
  #     print(v[x,y],vintr(x,y))
  for i in range(n) :
    print(i),print('i')
    # la
    z = 0
    while z < t :
      P1 = np.ones([3, 20*t])
      x = np.random.randint(u.shape[1], size=[1, 20*t])
      y = np.random.randint(u.shape[0], size=[1, 20*t])
      P1[0,:] = x
      P1[1,:] = y
      P1, L1 = get_filter( P1, T, v.shape, u.shape)
      z = P1.shape[1] 
      # print(P1.shape),
      # print('shape')
    P1 = P1[:,:t]
    L1 = L1[:,:t]
    # lb 
    # print(P1.shape)
    z = 0
    while z < t :
      P2 = np.ones([3, 20*t])
      x = np.random.randint(u.shape[1], size=[1, 20*t])
      y = np.random.randint(u.shape[0], size=[1, 20*t])
      P2[0,:] = x
      P2[1,:] = y
      P2, L2 = get_filter( P2, T, v.shape, u.shape)
      z = P2.shape[1]
    P2 = P2[:,:t] 
    L2 = L2[:,:t]
    if i==n-1 and __debug:
      # print(T)
      x = np.asarray([T[:,0]]).T@(np.asarray([np.tile(range(u.shape[1]), (u.shape[0], 1)).flatten()]) - (u.shape[1]/2.0))
      y = np.asarray([T[:,1]]).T@(np.asarray([np.tile(np.array([range(u.shape[0])]).T, (1, u.shape[1])).flatten()]) - (u.shape[0]/2.0))
      q = np.asarray([T[:,2]]).T@np.ones((1,u.shape[0]*u.shape[1]))
      x = x + y + q + np.tile(np.asarray([[v.shape[1]/2.0, v.shape[0]/2.0, 0]]).T, [1, u.shape[0]*u.shape[1]]) 
      v1=np.zeros(u.shape[0]*u.shape[1])
      for u2 in range(0,u.shape[0]*u.shape[1]):
        v1[u2]=vintr(x[1,u2], x[0,u2])
      v1 = v1.reshape(v.shape)
      print(v1.shape)
      plt.imshow(v1, norm=colr.Normalize().autoscale(v1));
      plt.show()
      # plt.pause()

# TransformDeriv(T,u,la,lb,Tla,Tlb,CovInvU,CovInvV,Vinterp,gradVx,gradVy)
    
    Tn = TransformDeriv(T, u, v, P1.astype(np.int16), P2.astype(np.int16),L1,L2, 100, 100, vintr, gradV[1], gradV[0])  
    # print(Tn)
    T[0, :] = T[0, :] + (alp* Tn[0,:])
    T[1, :] = T[1, :] + (alp* Tn[1,:])
    if __debug :
      print(T)
  return T 
