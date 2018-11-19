import numpy as np
def TransformDeriv(T,u,v,la,lb,CovInvU,CovInvV):
	Va=np.zeros((len(la),1))
	Vb=np.zeros((1,len(lb)))
	Ua=np.zeros((len(la),1))
	Ub=np.zeros((1,len(lb)))
	for x in range(0,len(la)):
		Va[x,0] = v[la[x][0],la[x][1]]
		Ua[x,0] = u[la[x][0],la[x][1]]
	for x in range(0,len(lb)):
		Vb[0,x] = v[lb[x][0],lb[x][1]]
		Ub[0,x] = u[lb[x][0],lb[x][1]]
	Va=np.tile(Va,(1,len(lb)))
	Vb=np.tile(Vb,(len(la),1))
	Ua=np.tile(Ua,(1,len(lb)))
	Ub=np.tile(Ub,(len(la),1))
	Wa = np.stack((Ua,Va))
	Wb = np.stack((Ub,Vb))
	ex=Wb-Wa
	CovUV=np.tile(np.array([CovInvU,CovInvV]).reshape(2,1,1),np.shape(Ua))
	G1=np.exp((Vb-Va)*(Vb-Va)*CovInvV)
	G1=G1/np.sum(G1,0)
	G2=np.exp(np.sum(ex*ex*CovUV,0))
	deriv=np.ones((len(la),len(lb)))
	p=(G1-G2)*(Vb-Va)*deriv
	np.sum(p)
v = np.arange(10000).reshape(100,100)/1000
u=v*2
# print(v)
la=[[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2],[2,3],[1,4],[0,2]]
lb=[[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0],[4,1],[2,0]]
T=np.zeros(10)
print(len(lb))
for x in range(1,1000):
	print(x)
	TransformDeriv(T,u,v,la,lb,.03,.03)