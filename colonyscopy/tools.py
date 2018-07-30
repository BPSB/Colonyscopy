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
