from itertools import product
from scipy.signal.windows import blackman
from scipy.signal import convolve2d
from scipy.stats import gaussian_kde
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

def color_sum(data):
	return np.sum(data, axis=-1)

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

def gaussian_profile(abscissae,position,width):
	profile = np.exp(-((abscissae-position)/width)**2/2)
	return profile/sum(profile)

def radial_profile(data,centre,smooth_width=0.5,npoints=100):
	radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None] ) )
	r_max = np.max(radii)
	abscissae = np.linspace(0,r_max,npoints)
	sums = np.zeros_like(abscissae)
	normalisation = np.zeros_like(abscissae)
	for i,radii_line in enumerate(radii):
		for j,radius in enumerate(radii_line):
			profile = gaussian_profile(abscissae,radius,smooth_width)
			normalisation += profile
			sums += profile*data[i,j]
	return abscissae,sums/normalisation

def excentricity(data,centre,smooth_width=0.5,npoints=100):
	abscissae,values = radial_profile(data,centre,smooth_width=2,npoints=100)
	radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None] ) )
	
	sumsq = 0
	for i,line in enumerate(data):
		for j,intensity in enumerate(line):
			expected_intensity = np.interp(radii[i,j],abscissae,values)
			sumsq += (intensity - expected_intensity)**2
	return np.sqrt(sumsq)/data.size

def new_excentricity(data,centre,width=0.1,cutoff=5,npoints=10):
	radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None] ) )
	r_max = np.max(radii)
	interval = [ cutoff, r_max-cutoff ]
	abscissae = np.linspace(*interval,npoints)
	mask = np.logical_and( interval[0]<radii, radii<interval[1] )
	
	kernel = gaussian_kde( radii.flatten(), bw_method=width, weights=data.flatten() )
	normalisation = gaussian_kde( radii.flatten(), bw_method=width )
	values = kernel.evaluate(abscissae)/normalisation.evaluate(abscissae)*np.average(data)
	
	expected_intensity = np.interp(radii[mask],abscissae,values)
	return np.sqrt(np.sum((data[mask]-expected_intensity)**2))/data.size

def circle_mask(resolution, centre, width):
    circle_mask = np.zeros(resolution, dtype=bool)
    circle_mask[tuple(centre)] = True
    circle_mask = expand_mask(circle_mask, width)
    return circle_mask
