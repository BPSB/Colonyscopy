from colonyscopy.tools import radial_profile, expand_mask
import numpy as np
from numpy.testing import assert_array_equal

resolution = (200,200)
centre = np.array([100,100])
radius = 20
x, y = np.indices(resolution)-centre[:,None,None]
distances = np.hypot(x,y)

class TestRadialProfile(object):
	def test_linear_profile(self):
		bins,values = radial_profile(distances,centre=centre)
		assert np.sum((values-bins)**2) < 2.0
	
	def test_gaußian_profile(self):
		function = lambda x: np.exp(-(x/10)**2)
		gaußian = function(distances)
		bins,values = radial_profile(gaußian,centre=centre)
		expected = function(bins)
		assert np.sum((values-expected)**2) < 2.0

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


