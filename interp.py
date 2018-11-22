from scipy import interpolate
import numpy as np
def interpolation(v):
	k=np.shape(v)[0]
	k1=np.shape(v)[1]
	return interpolate.interp2d(np.arange(k),np.arange(k1),v.T,kind='linear',fill_value=0)