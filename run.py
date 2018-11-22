import numpy as np
from numpy import pi
import numpy.ma as ma
from PIL import Image
from interp import interpolation
import matplotlib.pyplot as plt
import matplotlib.colors as colr
import sys

def readImages(u1,v1):
	try:
		u = np.asarray( Image.open(u1).convert('L'), dtype=np.float64)
		u=u/np.max(u)
		v = np.asarray( Image.open(v1).convert('L'), dtype=np.float64)
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
	# print(pdf1*pdf2)
	J=np.nan_to_num(JPDF/(pdf1*pdf2))
	# print(J)
	# exit()
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
	# print(Pv)
	vp=v[Pv[1],Pv[0]]
	up=u[Pu[1],Pu[0]]
	return (up,vp,Pv)
	# return points,tran

if( len(sys.argv) != 3) :
	print("usage: <" + sys.argv[0] + "> image_1 image_2\n")
	exit(1)
im_1 = sys.argv[1]
im_2 = sys.argv[2]

u,v=readImages(im_1,im_2)
vintr=interpolation(v)
# t=np.identity(3)
t=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
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
thet1=0
MI=np.zeros((9,9))
for i in range(-50,51,10):
	print(i)
	t[0][2]=i
	for j in range(-50,51,10):
		t[1][2]=j
		# print(t)
		for theta in range(-180,180,15):
			rad=pi*theta/180
			# print(np.array([[np.cos(rad),-1*np.sin(rad)],[np.sin(rad),np.cos(rad)]]))
			# print(t)
			t[0,0] = np.cos(rad)
			t[0,1] = -1*np.sin(rad)
			t[1,0] = np.sin(rad)
			t[1,1] = np.cos(rad)
			# print(t)
			up,vp,_=pointFilt(t,points,C1,C2)
			# print(vp)
			k=calcMI(up,vp,a,b)
			# MI[(j+20)//5,(i+20)//5]=k
			if(k>m):
				m=k
				i1=i
				j1=j
				thet1=theta
				t1=np.copy(t)
thet2=thet1
print(["m",m,"ijtheta",i1,j1,thet1])
for i in range(i1-5,i1+6,1):
	print(i)
	t[0][2]=i
	for j in range(j1-5,j1+6,1):
		t[1][2]=j
		for theta in range(thet1-7,thet1+8,1):
			rad=pi*theta/180
			t[0,0] = np.cos(rad)
			t[0,1] = -1*np.sin(rad)
			t[1,0] = np.sin(rad)
			t[1,1] = np.cos(rad)
			up,vp,_=pointFilt(t,points,C1,C2)
			k=calcMI(up,vp,a,b)
			# print([i,j,theta,k])
			# MI[(j+20)//5,(i+20)//5]=k
			if(k>m):
				m=k
				thet2=theta
				t1=np.copy(t)
	#T1 = np.linalg.inv(t)
print(t1)
print(thet2)
print(m)
x = np.matmul(np.asarray([t1[:,0]]).T,np.asarray([np.tile(range(u.shape[1]), (u.shape[0], 1)).flatten()]) - u.shape[1]/2.0)
y = np.matmul(np.asarray([t1[:,1]]).T,np.asarray([np.tile(np.array([range(u.shape[0])]).T, (1, u.shape[1])).flatten()]) - u.shape[0]/2.0)
q = np.matmul(np.asarray([t1[:,2]]).T,np.ones((1,u.shape[0]*u.shape[1])))
x = x + y + q + np.tile(np.array([[v.shape[1]/2.0, v.shape[0]/2.0, 0]]).T, [1,u.shape[0]*u.shape[1]])
v1=np.zeros(u.shape[0]*u.shape[1])
for u2 in range(0,u.shape[0]*u.shape[1]):
  v1[u2]=vintr(x[1,u2], x[0,u2])
v1 = v1.reshape(u.shape)
plt.imshow(v1,cmap='gray');
plt.show()
# print(MI)
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
