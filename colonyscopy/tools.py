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

def smoothen_image(image,width):
	kernel = blackman(width)[:,None]*blackman(width)
	kernel /= np.sum(kernel)
	result = np.empty_like(image)
	for i in range(3):
		result[:,:,i] = convolve2d(image[:,:,i],kernel,mode="same")
	return result

def expand_mask(mask,width):
	"""
	Expands `mask` (array of booleans) by setting all elments within a distance of `width` to a True element to True.
	"""
	centre = np.array([width,width])
	coordinates = np.indices((2*width+1,2*width+1))
	distances = np.linalg.norm(coordinates-centre[:,None,None],axis=0)
	kernel = (distances <= width).astype(float)
	return convolve2d(mask,kernel,mode="same").astype(bool)

def color_distance(data1,data2):
	"""
	Determines the L1 norm between the two input arrays in color space. The color axis must be the last axis.
	"""
	return np.sum(np.abs(np.subtract(data1,data2,dtype=float)),axis=-1)

class TestColorDistance(object):
	def test_symmetry(self):
		size = (200,100,3)
		x = np.random.randint(0,256,size=size,dtype=np.uint8)
		y = np.random.randint(0,256,size=size,dtype=np.uint8)
		np.testing.assert_array_equal(
				color_distance(x,y),
				color_distance(y,x),
			)

	def test_shape(self):
		size = (200,100,3)
		x = np.random.randint(0,256,size=size,dtype=np.uint8)
		y = np.random.randint(0,256,size=size,dtype=np.uint8)
		np.testing.assert_array_equal(color_distance(x,y).shape,size[:-1])

def show_image(array,mode="RGB"):
	img = Image.fromarray(array,mode)
	img.show()
	time.sleep(3)
	for proc in psutil.process_iter():
		if proc.name() == "display":
			proc.kill()

def radial_profile(data,centre,nbins=100,r_max=None):
	radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None] ) )
	r_max = r_max or np.max(radii)*(nbins+0.5)/nbins
	bin_centres = np.linspace(0,r_max,2*nbins+1)[1::2]
	bins = (radii*(nbins/r_max)).ravel().astype(int)
	values = np.bincount( bins, data.ravel() )
	normalisation = np.bincount(bins)
	return bin_centres,values/normalisation
