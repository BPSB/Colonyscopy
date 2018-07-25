from pytest import raises, approx
import numpy as np
from numpy.testing import assert_allclose
from colonyscopy import ColonyArray
from generate_data import generate_data,colony_centres

def test_segmenting():
	times = 10
	dimensions = (12,8)
	colony_sizes = np.dstack(
			np.random.uniform(i+1,4*(i+1),dimensions)
			for i in range(times)
		).transpose(2,0,1)
	resolution = (900,600)
	bg = np.random.randint(0,256,3)
	fg = np.random.randint(0,256,3)
	
	some_array = ColonyArray(
			images = generate_data(colony_sizes,resolution,bg,fg),
			dimensions = dimensions,
		#	bg = bg,
		)
	
	centres_control = colony_centres(dimensions,resolution)
	assert_allclose(some_array.centres,centres_control,atol=3)
	
	for i in (0,1):
		borders_control = np.linspace(0,resolution[i],dimensions[i]+1)
		assert_allclose(some_array.borders[i],borders_control,atol=3)
	
	# for i,j in [ (dimensions[0],0), (0,dimensions[1]) ]:
	# 	with raises(IndexError):
	# 		some_array[i,j]



