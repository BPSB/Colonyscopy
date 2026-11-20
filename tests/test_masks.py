from pytest import mark
import pytest
import numpy as np
from colonyscopy._colonyscopy import Plate
from colonyscopy.tools import expand_mask

size = (300,200)
times = 5
ncolours = 3


@pytest.fixture
def dummy_plate():
	plate = Plate(
		images=np.zeros((times,*size,ncolours)),
		bg=np.zeros((*size,ncolours)),
		layout=(2,2)
	)
	plate._coordinates = [np.array([75,225]),np.array([50,150])]
	return plate


@mark.parametrize("apply_mask,needs_expansion",[
	(lambda plate: plate._gradient_mask(factor=4),True),
	(lambda plate: plate._intensity_mask(factor=4),False),
])
class TestMasks(object):

	def test_format(self,apply_mask,needs_expansion,dummy_plate):
		mask = apply_mask(dummy_plate)

		assert mask.shape == size
		assert mask.dtype == bool

	def test_colour_independence(self,apply_mask,needs_expansion):
		colours = np.array([50,100,150])
		plate = Plate(
			images=np.broadcast_to(colours,(times,*size,ncolours)).copy(),
			bg=np.broadcast_to(colours,(*size,ncolours)).copy()
		)

		mask = apply_mask(plate)

		assert not np.any(mask)

	def test_single_changed_pixel(self,apply_mask,needs_expansion):
		point = (150,100)
		background = np.zeros((*size,ncolours))
		background[point[0],point[1],0] = 100

		plate = Plate(
			images=np.zeros((times,*size,ncolours)),
			bg=background
		)

		mask = apply_mask(plate)

		if needs_expansion:
			mask = expand_mask(mask,1)

		assert mask[point]

		if needs_expansion:
			far_from_point = ~expand_mask(background[:,:,0]>0,2)
			assert not np.any(mask[far_from_point])
		else:
			assert np.sum(mask) == 1


class TestTemporalMask(object):

	def test_format(self,dummy_plate):
		mask = dummy_plate._temporal_mask(radius=10,safety_factor=1.05)

		assert mask.shape == size
		assert mask.dtype == bool

	def test_no_temporal_changes(self,dummy_plate):
		mask = dummy_plate._temporal_mask(radius=10,safety_factor=1.05)

		assert not np.any(mask)

	def test_detects_large_speckle(self,dummy_plate):
		colony_point = (75,50)
		speckle_point = (150,100)

		dummy_plate.images[2:,70:81,45:56,0] = 100
		dummy_plate.images[2:,speckle_point[0],speckle_point[1],0] = 1000

		mask = dummy_plate._temporal_mask(radius=10,safety_factor=1.05)

		assert mask[speckle_point]
		assert not mask[colony_point]
		assert not mask[0,0]


class TestSpeckleMask(object):

	def test_format(self,dummy_plate):
		dummy_plate.create_speckle_mask()
		mask = dummy_plate._speckle_mask

		assert mask.shape == size
		assert mask.dtype == bool

	def test_no_speckles(self,dummy_plate):
		dummy_plate.create_speckle_mask()

		assert np.all(dummy_plate._speckle_mask)

	def test_single_speckle(self,dummy_plate):
		colony_point = (75,50)
		speckle_point = (150,100)

		dummy_plate.images[2:,70:81,45:56,0] = 100
		dummy_plate.images[2:,speckle_point[0],speckle_point[1],0] = 2000

		dummy_plate.create_speckle_mask(
			temporal_radius=10,
			temporal_safety_factor=1.05,
			expansion=2
		)

		mask = dummy_plate._speckle_mask

		assert not mask[speckle_point]
		assert not mask[150,101]
		assert mask[colony_point]
		assert mask[0,0]

	def test_recreates_mask(self,dummy_plate):
		dummy_plate.create_speckle_mask()
		first_mask = dummy_plate._speckle_mask

		dummy_plate.create_speckle_mask()
		second_mask = dummy_plate._speckle_mask

		assert first_mask is not second_mask
		np.testing.assert_array_equal(first_mask,second_mask)

	def test_combines_all_masks(self,dummy_plate):
		images = dummy_plate.images
		background = dummy_plate.background

		gradient_point = (50,50)
		intensity_point = (150,100)
		temporal_point = (250,150)
		colony_point = (75,50)

		background[50:60,50:60,:] = 1000
		background[intensity_point] = 1000
		images[2:,colony_point[0],colony_point[1],0] = 100
		images[2:,temporal_point[0],temporal_point[1],0] = 2000

		dummy_plate.create_speckle_mask(
			gradient_factor=4,
			intensity_factor=4,
			temporal_radius=10,
			temporal_safety_factor=1.05,
			expansion=0
		)

		expected = ~(
			dummy_plate._gradient_mask(factor=4)
			| dummy_plate._intensity_mask(factor=4)
			| dummy_plate._temporal_mask(radius=10,safety_factor=1.05)
		)

		mask = dummy_plate._speckle_mask

		np.testing.assert_array_equal(mask,expected)

		assert not mask[gradient_point]
		assert not mask[intensity_point]
		assert not mask[temporal_point]
		assert mask[0,0]

	@mark.parametrize("method",["gradient","intensity","temporal"])
	def test_individual_detection_methods(self,dummy_plate,method):
		kwargs = {
			"use_gradient":method=="gradient",
			"use_intensity":method=="intensity",
			"use_temporal":method=="temporal",
			"expansion":0
		}

		dummy_plate.create_speckle_mask(**kwargs)

		if method == "gradient":
			expected = ~dummy_plate._gradient_mask()
		elif method == "intensity":
			expected = ~dummy_plate._intensity_mask()
		else:
			expected = ~dummy_plate._temporal_mask()

		np.testing.assert_array_equal(dummy_plate._speckle_mask,expected)

	def test_all_methods_disabled(self,dummy_plate):
		dummy_plate.create_speckle_mask(
			use_gradient=False,
			use_intensity=False,
			use_temporal=False
		)

		assert np.all(dummy_plate._speckle_mask)


def test_gradient_mask():
	images = np.zeros((2,20,20,3))
	background = np.zeros((20,20,3))
	background[:,10:,:] = 100

	plate = Plate(images=images,bg=background)
	mask = plate._gradient_mask()

	assert np.any(mask[:,9:11])
	assert not np.any(mask[:,:8])
	assert not np.any(mask[:,12:])


class TestColonyReferenceMask(object):

	def test_reference_regions(self,dummy_plate):
		mask = dummy_plate._colony_reference_mask(radius=10)

		assert mask.shape == size
		assert mask.dtype == bool

		for row,col in dummy_plate.centres.reshape(-1,2):
			assert mask[int(row),int(col)]

		assert not mask[150,100]
		assert not mask[0,0]
