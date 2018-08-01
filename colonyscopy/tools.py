from itertools import product
from scipy.signal.windows import blackman
from scipy.signal import convolve2d
import numpy as np
from PIL import Image
import time
import psutil

def smoothen(data,width):
	kernel = blackman(width)
	kernel /= np.sum(kernel)
	return np.convolve(data,kernel,mode="same")

def smoothen_image(data,width):
	kernel = blackman(width)[:,None]*blackman(width)
	kernel /= np.sum(kernel)
	result = np.empty_like(data)
	for i in range(3):
		result[:,:,i] = convolve2d(data[:,:,i],kernel,mode="same")
	return result

def color_distance(data1,data2):
	"""
	Determines the L1 norm between the two input arrays in color space. The color axis must be the last axis.
	"""
	assert data1.shape[-1]==3
	return np.sum(np.abs(data1-data2),axis=-1)

def show_image(array):
	img = Image.fromarray(array,"RGB")
	img.show()
	time.sleep(3)
	for proc in psutil.process_iter():
		if proc.name() == "display":
			proc.kill()

def radial_profile(data,centre,nbins=100):
	r_max= max(
			np.hypot(corner,centre)
			for corner in product(*zip((0,0),data.shape))
		)
	
	coordinates = np.indices(data.shape).transpose(1,2,0)
	radii = np.linalg.norm( coordinates-centre, axis=2 )
	bins = np.minimum((radii*nbins/r_max).astype(int),nbins)
	normalisation = np.zeros(nbins,dtype=int)
	summe = np.zeros(nbins)
	
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			normalisation[bins[i,j]] += 1
			summe[bins[i,j]] += data[i,j]
	
	return summe/normalisation
	


