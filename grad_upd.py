import numpy as np
from scipy import interpolate
# T => 3x3 transform matrix
# u => first image
# la => coordinates for parzen window 2xNa
# lb => coordinates for entropy calculation 2xNb
# Tla => coordinates for parzen window 2xNa
# Tlb => coordinates for entropy calculation 2xNb
# CovInvU => float
# CovInvV => float
# Vinterp => interpolate object for v
# gradVx => interpolate object gradient along x for matrix v
# gradVy => interpolate object gradient along y for matrix v
def TransformDeriv(T,u,la,lb,CovInvU,CovInvV,Vinterp,gradVx,gradVy):
	Na = np.shape(la)[1]
	Nb = np.shape(lb)[1]
	Va=np.zeros((Na,1))
	Vb=np.zeros((1,Nb))
	dVa=np.zeros((2,Na)) # gradients for set A points 2XNa
	dVb=np.zeros((2,Nb)) # gradients for set B points 2XNb
	Ua=np.zeros((Na,1))
	Ub=np.zeros((1,Nb))
	for x in range(0,Na):
		Va[x,0] = Vinterp(Tla[0][x],Tla[1][x])
		dVa[0,x] = gradVx(Tla[0][x],Tla[1][x])
		dVa[1,x] = gradVy(Tla[0][x],Tla[1][x])
		Ua[x,0] = u[la[0][x],la[1][x]]
	for x in range(0,Nb):
		Vb[0,x] = Vinterp(Tlb[0][x],Tlb[1][x])
		dVb[0,x] = gradVx(Tla[0][x],Tla[1][x])
		dVb[1,x] = gradVy(Tla[0][x],Tla[1][x])
		Ub[0,x] = u[lb[0][x],lb[1][x]]
	Va=np.tile(Va,(1,Nb)) # replicate columns of Va to make it NaxNb
	Vb=np.tile(Vb,(Na,1)) # replicate rows of Vb to make it NaxNb
	Ua=np.tile(Ua,(1,Nb)) # same as V
	Ub=np.tile(Ub,(Na,1)) # same as V
	dVa=dVa.reshape(Na,2,1) # make the gradient 2x1 with depth Na
	dVb=dVb.reshape(Nb,2,1) # make the gradient 2x1 with depth Nb
	xTa = Tla.reshape(Na,1,3) # make the x' 1x3 with depth Na
	xTb = Tlb.reshape(Nb,1,3) # make the x' 1x3 with depth Nb
	dTa=dVa@xTa # each gradient vector is multiplied with x' resulting in 2x3 matrix with depth Na
	dTb=dVb@xTb # each gradient vector is multiplied with x' resulting in 2x3 matrix with depth Nb
	dTa=dTa.flatten().reshape(Na,6).T.reshape(6,Na,1) # reshape to Nax6 and then again reshape to make dT along depth
	dTa=np.tile(dTa,(1,1,Nb))
	dTb=dTb.flatten().reshape(Nb,6).T.reshape(6,1,Nb)
	dTb=np.tile(dTb,(1,Na,1))
	deriv=dTb - dTa # derivative of vi-vj in each cell with 6 elements along the depth i vary along rows j along columns
	Wa = np.stack((Ua,Va))
	Wb = np.stack((Ub,Vb))
	ex=Wb-Wa
	CovUV=np.tile(np.array([CovInvU,CovInvV]).reshape(2,1,1),np.shape(Ua))
	G1=np.exp((Vb-Va)*(Vb-Va)*CovInvV)
	G1=G1/np.sum(G1,0) # W_v(v_i,v_j)*
	G2=np.exp(np.sum(ex*ex*CovUV,0))
	# deriv=np.array()
	p=(G1*CovInvV-G2*CovInvV)*(Vb-Va)*deriv
	return np.sum(p,(1,2)).reshape(2,3)
# v = np.arange(10000).reshape(100,100)/1000
# u=v*2
# # print(v)
# la=[[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2]]
# lb=[[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0]]
# T=np.zeros(9).reshape(3,3)
# print(len(lb))
# for x in range(1,1000):
# 	print(x)
# 	TransformDeriv(T,u,v,la,lb,.03,.03)
