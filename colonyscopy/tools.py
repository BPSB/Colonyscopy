from scipy.signal.windows import blackman
import numpy as np

def smoothen(data,width):
	kernel = blackman(width)
	kernel /= np.sum(kernel)
	return np.convolve(data,kernel,mode="same")

def color_distance(data1,data2):
	"""
	Determines the L1 norm between the two input arrays in color space. The color axis must be the last axis.
	"""
	assert data1.shape[-1]==3
	return np.sum(np.abs(data1-data2),axis=-1)

