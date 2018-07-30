from pytest import raises, approx, mark
import numpy as np
from numpy.testing import assert_allclose
from colonyscopy import Plate
from generate_data import generate_data,colony_centres
from colonyscopy.tools import show_image

times = 10
layout = (12,8)
colony_sizes = np.dstack(
		np.random.uniform(i+1,4*(i+1),layout)
		for i in range(times)
	).transpose(2,0,1)
resolution = (900,600)
bg,fg,rim_color = np.random.randint(0,256,(3,3))
rim_width = 5

@mark.parametrize(
		"noise_amp, tol",[
		(    0    ,  3 ),
		(   30    ,  3 ),
	])
def test_segmenting(noise_amp,tol):
	some_array = Plate(
			images = generate_data(
					colony_sizes,resolution,
					bg,fg,
					noise_amp,
					rim_width,rim_color
				),
			layout = layout,
		#	bg = bg,
		)
	
	# for image in some_array.images:
	# 	show_image(image)
	
	centres_control = colony_centres(layout,resolution)
	assert_allclose(some_array.centres,centres_control,atol=3)
	
	for i in (0,1):
		borders_control = np.linspace(0,resolution[i],layout[i]+1)
		assert_allclose(some_array.borders[i],borders_control,atol=tol)
	
	# for i,j in [ (layout[0],0), (0,layout[1]) ]:
	# 	with raises(IndexError):
	# 		some_array[i,j]



