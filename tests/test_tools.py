from colonyscopy.tools import radial_profile, expand_mask, excentricity
import numpy as np
from numpy.testing import assert_array_equal

resolution = (100,100)
centre = np.array([50,50])
radius = 20
x, y = np.indices(resolution)-centre[:,None,None]
distances = np.hypot(x,y)
gauß_function = lambda x: np.exp(-(x/radius)**2)
gaußian = gauß_function(distances)

class TestRadialProfile(object):
	def test_linear_profile(self):
		bins,values = radial_profile(distances,centre=centre)
		assert np.sum((values-bins)**2) < 2.0
	
	def test_gaußian_profile(self):
		bins,values = radial_profile(gaußian,centre=centre)
		expected = gauß_function(bins)
		assert np.sum((values-expected)**2) < 2.0

def assert_increasing(data):
	assert np.all(np.array(data[1:])-np.array(data[:-1]))

class TestExcentricity(object):
	def test_single_point(self):
		original_excentricity = excentricity(gaußian,centre=centre,npoints=10)
		assert original_excentricity<1e-4
		
		gaußian_with_spot = gaußian.copy()
		gaußian_with_spot[0,0] = 10*np.max(gaußian_with_spot)
		excentricity_with_spot = excentricity(gaußian_with_spot,centre=centre,npoints=10)
		assert excentricity_with_spot>original_excentricity
	
	def test_distortion_dependence(self):
		eccentricities = []
		for strech_factor in 10**np.linspace(0,1,10):
			distorted_distances = np.hypot(x,strech_factor*y)
			distorted_gaußian = gauß_function(distorted_distances)
			eccentricities.append(excentricity(distorted_gaußian,centre=centre,npoints=10))
		
		assert_increasing(eccentricities)
	
	def test_offset_dependence(self):
		eccentricities = [
			excentricity(gaußian,centre=centre+[offset,0],npoints=10)
			for offset in np.linspace(0,1,10)
		]
		
		assert_increasing(eccentricities)

class TestExpandMask(object):
	def test_zero_radius(self):
		random_mask = np.random.choice([True,False],size=(100,80))
		assert_array_equal( expand_mask(random_mask,0), random_mask )
	
	def test_points(self):
		size = (100,80)
		points = np.vstack([
				np.random.randint(n,size=5)
				for n in size
			]).T
		radius = 13
		mask = np.zeros(size,dtype=bool)
		for point in points:
			mask[tuple(point)] = True
		expanded_mask = expand_mask(mask,radius)
		for i in range(size[0]):
			for j in range(size[1]):
				dist = min(
						np.linalg.norm(point-[i,j])
						for point in points
					)
				assert (dist<=radius) == expanded_mask[i,j]


