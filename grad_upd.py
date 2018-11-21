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
def TransformDeriv(T,u,la,lb,Tla,Tlb,CovInvU,CovInvV,Vinterp,gradVx,gradVy):
	Na = np.shape(la)[1]
	Nb = np.shape(lb)[1]
	# print(Na)
	# print(Nb)
	Va=np.zeros((Na,1))
	Vb=np.zeros((1,Nb))
	dVa=np.zeros((2,Na)) # gradients for set A points 2XNa
	dVb=np.zeros((2,Nb)) # gradients for set B points 2XNb
	Ua=np.zeros((Na,1))
	Ub=np.zeros((1,Nb))
	for x in range(0,Na):
		Va[x,0] = Vinterp(Tla[1,x],Tla[0,x])
		dVa[0,x] = gradVx(Tla[1,x],Tla[0,x])
		dVa[1,x] = gradVy(Tla[1,x],Tla[0,x])
		Ua[x,0] = u[la[1,x],la[0,x]]
	for x in range(0,Nb):
		Vb[0,x] = Vinterp(Tlb[1,x],Tlb[0,x])
		dVb[0,x] = gradVx(Tlb[1,x],Tlb[0,x])
		dVb[1,x] = gradVy(Tlb[1,x],Tlb[0,x])
		Ub[0,x] = u[lb[1,x],lb[0,x]]
	Va=np.tile(Va,(1,Nb)) # replicate columns of Va to make it NaxNb
	# print(Va)
	Vb=np.tile(Vb,(Na,1)) # replicate rows of Vb to make it NaxNb
	# print(Vb)
	Ua=np.tile(Ua,(1,Nb)) # same as V
	# print(Ua)
	Ub=np.tile(Ub,(Na,1)) # same as V
	# print(Ub)
	dVa=dVa.T.reshape(Na,2,1) # make the gradient 2x1 with depth Na
	# print(dVa)
	dVb=dVb.T.reshape(Nb,2,1) # make the gradient 2x1 with depth Nb
	# print(dVa)
	xTa = Tla.T.reshape(Na,1,3) # make the x' 1x3 with depth Na
	# print(xTa)
	xTb = Tlb.T.reshape(Nb,1,3) # make the x' 1x3 with depth Nb
	# print(xTb)
	dTa=dVa@xTa # each gradient vector is multiplied with x' resulting in 2x3 matrix with depth Na
	dTb=dVb@xTb # each gradient vector is multiplied with x' resulting in 2x3 matrix with depth Nb
	dTa=dTa.flatten().reshape(Na,6).T.reshape(6,Na,1) # reshape to Nax6 and then again reshape to make dT along depth
	dTa=np.tile(dTa,(1,1,Nb))
	dTb=dTb.flatten().reshape(Nb,6).T.reshape(6,1,Nb)
	# print(dTb)
	dTb=np.tile(dTb,(1,Na,1))
	# print(dTb)
	deriv=dTb - dTa # derivative of vi-vj in each cell with 6 elements along the depth i vary along rows j along columns
	# print(Ua)
	# print(Va)
	Wa = np.stack((Ua,Va))
	# print(Wa)
	Wb = np.stack((Ub,Vb))
	ex=Wb-Wa
	# print(Wa - Wb)
	CovUV=np.tile(np.array([CovInvU,CovInvV]).reshape(2,1,1),np.shape(Ua))
	# print(ex*ex*CovUV)
	G1=np.exp((-1/2)*(Vb-Va)*(Vb-Va)*CovInvV)
	# print(G1)
	G1=G1/np.sum(G1,0) # W_v(v_i,v_j)*
	G2=np.exp((-1/2)*np.sum(ex*ex*CovUV,0))
	# print(G2)
	G2=G2/np.sum(G2,0) # W_v(v_i,v_j)*
	# deriv=np.array()
	p=(G1*CovInvV-G2*CovInvV)*(Vb-Va)*deriv
	return np.sum(p,(1,2)).reshape(2,3)/Nb
