from pytest import raises, approx, mark
import numpy as np
from numpy.testing import assert_allclose
from colonyscopy import Plate
from generate_data import generate_data,colony_centres
from colonyscopy.tools import show_image, smoothen_image

times = 10
layout = (12,8)
colony_sizes = np.dstack(
		np.random.uniform(i+1,4*(i+1),layout)
		for i in range(times)
	).transpose(2,0,1)
resolution = (900,600)
bg,fg,rim_color = np.random.randint(0,256,(3,3))
rim_width = 5
noise_width = 20

bg_img = np.empty((*resolution,3),dtype=np.uint8)
for c in range(3):
	middle = bg[c]//2+fg[c]//2
	interval = sorted((bg[c],middle))
	bg_img[:,:,c] = np.random.randint(*interval,resolution)
bg_img = smoothen_image(bg_img,20)

@mark.parametrize(
		"noise_amp, tol",[
		(    0    ,  3 ),
		(   30    ,  3 ),
	])
@mark.parametrize("background",[bg,bg_img])
def test_segmenting(noise_amp,tol,background):
	some_array = Plate(
			images = generate_data(
					colony_sizes,resolution,
					background, fg,
					noise_amp, noise_width,
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



