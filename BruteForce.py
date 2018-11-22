import numpy as np
import numpy.ma as ma
from PIL import Image
from interp import interpolation
import matplotlib.pyplot as plt
import matplotlib.colors as colr
def readImages(u1,v1):
	try:
		u = np.asarray( Image.open(u1), dtype=np.float64)
		u=u/np.max(u)
		v = np.asarray( Image.open(v1), dtype=np.float64)
		v=v/np.max(v)
		return (u,v)
	except IOError:
		return -1
def calcMI(up,vp,a,b):
	MI=0
	H,xedge,yedge = np.histogram2d(up,vp,[a,b])
	JPDF = H/np.sum(H)
	pdf1=np.asarray([np.sum(JPDF,1)]).T
	pdf2=np.asarray([np.sum(JPDF,0)])
	pdf1=np.tile(pdf1,(1,b))
	pdf2=np.tile(pdf2,(a,1))
	J=JPDF/(pdf1*pdf2)
	J=ma.log(J)
	J=JPDF*J
	MI=np.sum(J)
	return MI
def pointFilt(t,points,C1,C2):
	pointsn=points-C1
	tran = np.matmul(t,pointsn)
	tran =tran + C2
	mask = np.logical_and(np.logical_and(tran[0,:] < C1[0]*2,tran[0,:] >= 0) , np.logical_and(tran[1,:] < C1[1]*2,tran[1,:] >= 0))
	# print(mask)
	Pu=(points.T[mask]).T
	Pv=(tran.T[mask]).T
	Pu=np.asarray(Pu[:2],dtype=np.int)
	Pv=np.asarray(Pv[:2],dtype=np.int)
	vp=v[Pv[1],Pv[0]]
	up=u[Pu[1],Pu[0]]
	return (up,vp)
	# return points,tran

im_1 = 'TestImages/test_mri.jpg'
# im_2 = 'TestImages/test_mri.jpg'
im_2 = 'TestImages/test_mri_translate2.jpg'
# im_2 = 'TestImages/test_mri_rot2.jpg'
# im_2 = 'TestImages/test_mri_rot.jpg'
# im_2 = 'TestImages/test_mri_scale.jpg'
# im_2 = 'TestImages/test_mri_shear.jpg'
u,v=readImages(im_1,im_2)
vintr=interpolation(v)
# t=np.identity(3)
t=np.array([[1,0,0],[0,1,0],[0,0,1]])
C1=np.asarray([u.T.shape]).T/2.0
C1=np.vstack((C1,np.zeros((1,1))))
C2=np.asarray([u.T.shape]).T/2.0
C2=np.vstack((C2,np.zeros((1,1))))
pointy = np.tile(np.asarray([np.arange(u.shape[0])]).T,(1,u.shape[1]))
pointx = np.tile(np.asarray([np.arange(u.shape[1])]),(u.shape[0],1))
ones = np.ones(u.shape)
points = np.stack((pointx,pointy,ones)).reshape(3,u.shape[0]*u.shape[1])
a=101
b=101
m=0
t1=0
i1=0
j1=0
MI=np.zeros((9,9))
for i in range(-20,21,5):
	t[0][2]=i
	for j in range(-20,21,5):
		t[1][2]=j
		up,vp=pointFilt(t,points,C1,C2)
		k=calcMI(up,vp,a,b)
		MI[(j+20)//5,(i+20)//5]=k
		if(k>m):
			m=k
			t1=np.copy(t)
	#T1 = np.linalg.inv(t)
x = np.matmul(np.asarray([t1[:,0]]).T,np.asarray([np.tile(range(u.shape[1]), (u.shape[0], 1)).flatten()]) - u.shape[1]/2.0)
y = np.matmul(np.asarray([t1[:,1]]).T,np.asarray([np.tile(np.array([range(u.shape[0])]).T, (1, u.shape[1])).flatten()]) - u.shape[0]/2.0)
q = np.matmul(np.asarray([t1[:,2]]).T,np.ones((1,u.shape[0]*u.shape[1])))
x = x + y + q + np.tile(np.array([[v.shape[1]/2.0, v.shape[0]/2.0, 0]]).T, [1,u.shape[0]*u.shape[1]])
v1=np.zeros(u.shape[0]*u.shape[1])
for u2 in range(0,v.shape[0]*v.shape[1]):
  v1[u2]=vintr(x[1,u2], x[0,u2])
v1 = v1.reshape(v.shape)
print(v1.shape)
plt.imshow(v1,cmap='gray');
plt.show()
print(MI)
H,xedge,yedge = np.histogram2d(u.flatten(),v.flatten(),[a,b])
JPDF = H/np.sum(H)
pdf1=np.asarray([np.sum(JPDF,1)]).T
pdf2=np.asarray([np.sum(JPDF,0)])
pdf1=np.tile(pdf1,(1,b))
pdf2=np.tile(pdf2,(a,1))
J=JPDF/(pdf1*pdf2)
J=ma.log(J)
J=JPDF*J
# MI=np.sum(J)
print(np.sum(J))
# for x in range(0,1):
# # print(np.min(pdf1*pdf2))
# # print(pdf2)
# # print(JPDF)
# # print(xedge)
