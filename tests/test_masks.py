from pytest import mark
import numpy as np
from colonyscopy import Plate
from colonyscopy.tools import expand_mask

size = (300,200)
times = 5
ncolours = 4

@mark.parametrize("apply_mask, needs_expansion",[
		((lambda plate: plate._gradient_mask(threshold=1e-5)), True),
		((lambda plate: plate._intensity_mask(factor=4)), False),
		((lambda plate: plate._temporal_mask(threshold=1e-5)), False),
	])
class TestMasks(object):
	def test_format(self,apply_mask,needs_expansion):
		dummy_plate = Plate(
				images = np.empty([times,*size,ncolours]),
				bg = np.empty((*size,ncolours)),
			)
		mask = apply_mask(dummy_plate)
		assert mask.shape == size
		assert mask.dtype == bool
	
	def test_colour_independence(self,apply_mask,needs_expansion):
		colours = np.random.randint(0,256,ncolours)
		dummy_plate = Plate(
				images = colours*np.ones([times,*size,ncolours],dtype=np.uint8),
				bg = colours*np.ones((*size,1),dtype=np.uint8),
			)
		mask = apply_mask(dummy_plate)
		assert not np.any(mask)
	
	def test_point(self,apply_mask,needs_expansion):
		point = tuple([np.random.randint(x) for x in size])
		colour = np.random.randint(ncolours)
		background = np.zeros((*size,ncolours),dtype=np.uint8)
		background[(*point,colour)] = 1
		dummy_plate = Plate(
				images = np.zeros([times,*size,ncolours]),
				bg = background,
			)
		
		mask = apply_mask(dummy_plate)
		if needs_expansion:
			mask = expand_mask(mask,1)
		
		assert mask[point]
		
		if needs_expansion:
			far_from_point = np.logical_not(expand_mask(background[:,:,colour]>0,2))
			assert not np.any(mask[far_from_point])
		else:
			assert np.sum(mask)==1


class TestTemporalMask(object):
	def test_point(self):
		point = tuple([np.random.randint(x) for x in size])
		colour = np.random.randint(ncolours)
		time = np.random.randint(times)
		
		images = np.zeros([times,*size,ncolours])
		images[(time,*point,colour)] = 1
		
		dummy_plate = Plate(
				images = images,
				bg = np.zeros((*size,ncolours),dtype=np.uint8),
			)
		
		mask = dummy_plate._temporal_mask(threshold=1e-5)
		
		assert mask[point]
		assert np.sum(mask)==1

