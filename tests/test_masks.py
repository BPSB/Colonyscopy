from pytest import mark
import numpy as np
from colonyscopy import Plate
from colonyscopy.tools import expand_mask

size = (300,200)
times = 5
ncolours = 4
dummy_plate = Plate(
		images = np.zeros([times,*size,ncolours]),
		bg = np.zeros([*size,ncolours]),
	)

@mark.parametrize("apply_mask",[
		lambda plate: plate.gradient_mask(threshold=1e-5),
		lambda plate: plate.intensity_mask(factor=4),
	])
class TestMasks(object):
	def test_dimensions(self,apply_mask):
		background = np.random.randint(0,256,(*size,ncolours))
		dummy_plate.background = background
		mask = apply_mask(dummy_plate)
		assert mask.shape == size
	
	def test_colour_independence(self,apply_mask):
		background = np.random.randint(0,256,ncolours)*np.ones((*size,1),dtype=np.uint8)
		dummy_plate.background = background
		mask = apply_mask(dummy_plate)
		assert not np.any(mask)
	
	def test_point(self,apply_mask):
		point = tuple([np.random.randint(x) for x in size])
		colour = np.random.randint(ncolours)
		background = np.zeros((*size,ncolours),dtype=np.uint8)
		background[(*point,colour)] = 1
		
		dummy_plate.background = background
		mask = apply_mask(dummy_plate)
		
		assert expand_mask(mask,1)[point]
		far_from_point = np.logical_not(expand_mask(background[:,:,colour]>0,2))
		assert not np.any(mask[far_from_point])

