from colonyscopy.tools import radial_profile
import numpy as np

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

