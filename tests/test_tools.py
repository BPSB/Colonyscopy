import pytest
from colonyscopy.tools import (
		load_images,
		radial_profile,
		expand_mask,
		excentricity,
		color_distance,
		smoothen,
		smoothen_image,
		color_sum,
		gaussian_profile,
		circle_mask,
	)
import numpy as np
from numpy.testing import assert_array_equal
from unittest.mock import patch, MagicMock
from PIL import Image
import rawpy

resolution = (100,100)
data = np.ones(resolution)
centre = np.array([50,50])
radius = 20
x, y = np.indices(resolution)-centre[:,None,None]
distances = np.hypot(x, y)
gauß_function = lambda x: np.exp(-(x/radius)**2)
gaußian = gauß_function(distances)

class TestLoadImages:

	def test_load_images(self,tmp_path):
		paths = []
		for i in range(3):
			image = np.full((10,10,3),i*50,dtype=np.uint8)
			path = tmp_path/f"image_{i}.png"
			Image.fromarray(image).save(path)
			paths.append(path)

		result = load_images(paths)

		assert result.shape == (3,10,10,3)
		for i in range(3):
			np.testing.assert_array_equal(result[i],i*50)

	def test_preserves_path_order(self,tmp_path):
		paths = []
		for name,value in [("image_3.png",30),("image_1.png",10),("image_2.png",20)]:
			path = tmp_path/name
			Image.fromarray(np.full((5,5,3),value,dtype=np.uint8)).save(path)
			paths.append(path)

		result = load_images(paths)

		assert result[:,0,0,0].tolist() == [30,10,20]

	def test_mixed_image_formats(self,tmp_path):
		paths = []
		for name in ("image_1.png","image_2.tiff"):
			path = tmp_path/name
			Image.fromarray(np.full((5,5,3),100,dtype=np.uint8)).save(path)
			paths.append(path)

		result = load_images(paths)

		assert result.shape == (2,5,5,3)

	def test_no_images_provided(self):
		with pytest.raises(ValueError,match="No image files provided"):
			load_images([])

	def test_unsupported_format(self,tmp_path):
		path = tmp_path/"image.bmp"

		with pytest.raises(ValueError,match="Unsupported image format"):
			load_images([path])

	def test_different_image_dtypes(self,tmp_path):
		png_path = tmp_path/"image.png"
		tiff_path = tmp_path/"image.tiff"

		Image.fromarray(np.zeros((10,10),dtype=np.uint8)).save(png_path)
		Image.fromarray(np.zeros((10,10),dtype=np.uint16)).save(tiff_path)

		with pytest.raises(ValueError,match="Image shape or dtype mismatch"):
			load_images([png_path,tiff_path])

	def test_load_cr2_images(self,tmp_path,monkeypatch):
		path = tmp_path/"image.cr2"
		path.touch()
		expected = np.full((10,10,3),100,dtype=np.uint8)

		class FakeRaw:
			def __enter__(self):
				return self

			def __exit__(self,*args):
				pass

			def postprocess(self,output_bps,use_camera_wb):
				assert output_bps == 8
				assert use_camera_wb is True
				return expected

		monkeypatch.setattr(rawpy,"imread",lambda path: FakeRaw())
		result = load_images([path])

		np.testing.assert_array_equal(result[0],expected)

	def test_cr2_decoding_error(self,tmp_path,monkeypatch):
		path = tmp_path/"image.cr2"
		path.touch()

		def fake_imread(path):
			raise rawpy.LibRawDataError("Invalid RAW file")

		monkeypatch.setattr(rawpy,"imread",fake_imread)

		with pytest.raises(RuntimeError,match="Failed to decode image 1/1"):
			load_images([path])

	def test_cr2_decoding_error_later_frame(self,tmp_path,monkeypatch):
		paths = [tmp_path/"image_1.cr2",tmp_path/"image_2.cr2"]
		for path in paths:
			path.touch()

		expected = np.zeros((10,10,3),dtype=np.uint8)

		class FakeRaw:
			def __enter__(self):
				return self

			def __exit__(self,*args):
				pass

			def postprocess(self,output_bps,use_camera_wb):
				return expected

		def fake_imread(path):
			if path.endswith("image_2.cr2"):
				raise rawpy.LibRawDataError("Invalid RAW file")
			return FakeRaw()

		monkeypatch.setattr(rawpy,"imread",fake_imread)

		with pytest.raises(RuntimeError,match="Failed to decode image 2/2"):
			load_images(paths)


class TestSmoothen(object):

	def test_output_shape(self):
		data = np.arange(100)
		result = smoothen(data, width=10)

		assert result.shape == data.shape

	def test_constant_data(self):
		data = np.ones(100)
		result = smoothen(data, width=10)

		#excludes edges affected by zero-padding
		np.testing.assert_allclose(result[10:-10], 1.0)

	def test_peak_reduction_through_smoothing(self):
		data = np.zeros(100)
		data[50] = 10
		result = smoothen(data, width=10)

		assert result[50] < data[50]
		assert result[50] > 0

class TestSmoothenImage(object):

	def test_output_shape(self):
		image = np.ones((100, 80, 3))
		result = smoothen_image(image, width=10)

		assert result.shape == image.shape

	def test_constant_image(self):
		image = np.ones((100, 80, 3)) * 100
		result = smoothen_image(image, width=10)

		#excludes edges affected by zero-padding
		np.testing.assert_allclose(result[10:-10, 10:-10], 100)

	def test_peak_reduction(self):
		image = np.zeros((100, 80, 3))
		image[50, 40, :] = 100
		result = smoothen_image(image, width=10)

		assert np.all(result[50, 40, :] < 100)
		assert np.all(result[50, 40, :] > 0)


class TestExpandMask(object):
	def test_zero_radius(self):
		random_mask = np.random.choice([True,False],size=(100,80))
		assert_array_equal( expand_mask(random_mask,0), random_mask )

	def test_expands_mask(self):
		size = (100,80)
		points = np.vstack([np.random.randint(n,size=5)for n in size]).T
		radius = 13
		mask = np.zeros(size,dtype=bool)
		for point in points:
			mask[tuple(point)] = True

		expanded_mask = expand_mask(mask,radius)

		for i in range(size[0]):
			for j in range(size[1]):
				dist = min(np.linalg.norm(point-[i,j])for point in points)

				assert (dist<=radius) == expanded_mask[i,j]


class TestColorSum(object):

	def test_color_sum(self):
		image = np.array([[[10, 20, 30], [40, 50, 60]],[[5, 10, 15], [20, 30, 40]]])
		result = color_sum(image)
		expected = np.array([[60, 150],[30, 90]])

		assert_array_equal(result, expected)

	def test_output_shape(self):
		image = np.ones((100, 80, 3))
		result = color_sum(image)

		assert result.shape == (100, 80)


class TestColorDistance(object):
	def test_symmetry(self):
		size = (200,100,3)
		x = np.random.randint(0,256,size=size,dtype=np.uint8)
		y = np.random.randint(0,256,size=size,dtype=np.uint8)

		np.testing.assert_array_equal(color_distance(x,y),color_distance(y,x),)

	def test_shape(self):
		size = (200,100,3)
		x = np.random.randint(0,256,size=size,dtype=np.uint8)
		y = np.random.randint(0,256,size=size,dtype=np.uint8)

		np.testing.assert_array_equal(color_distance(x,y).shape,size[:-1])

	def test_color_distance_values(self):
		x = np.array([[[10, 20, 30]]])
		y = np.array([[[20, 10, 50]]])
		result = color_distance(x, y)

		# absolute differences: 10 + 10 + 20 = 40
		assert_array_equal(result, [[40]])


class TestGaussianProfile(object):

	def test_normalisation(self):
		positions = np.linspace(0, 100, 100)
		result = gaussian_profile(positions, position=50, width=10)

		assert np.isclose(np.sum(result), 1.0)

	def test_symmetry(self):
		positions = np.arange(0, 101)
		result = gaussian_profile(positions, position=50, width=10)

		np.testing.assert_allclose(result, result[::-1])

	def test_maximum_at_centre(self):
		positions = np.arange(0, 101)
		result = gaussian_profile(positions, position=50, width=10)

		assert np.argmax(result) == 50

class TestRadialProfile(object):

	def test_radial_profile_output(self):
		bins, values = radial_profile(data, centre, npoints=25)

		assert bins.shape == (25,)
		assert values.shape == (25,)
		assert np.all(np.diff(bins) > 0)

	def test_linear_profile(self):
		bins,values = radial_profile(distances,centre=centre)

		assert np.sum((values-bins)**2)<2.0 #radial profile matches distance

	def test_gaußian_profile(self):
		bins,values = radial_profile(gaußian,centre=centre)
		expected = gauß_function(bins)
		assert np.sum((values-expected)**2)<2.0

	def test_speckle_mask(self):
		data = np.ones((20, 20))
		data[10, 10] = 100
		mask = np.zeros((20, 20), dtype=bool)
		mask[10, 10] = True
		bins, values = radial_profile(data,centre=(10, 10),speckle_mask=mask,npoints=20)

		np.testing.assert_allclose(values, 1.0)


def assert_increasing(data):
		assert np.all(np.diff(data) > 0)

class TestExcentricity(object):

	def test_circle_excentricity(self):
		result = excentricity(data, centre=centre, npoints=10)
		assert result < 1e-10 #uniform image should have very low excentricity

	def test_bright_spot_increases_excentricity(self):
		original_excentricity = excentricity(gaußian,centre=centre,npoints=60)
		assert original_excentricity<1e-4

		gaußian_with_spot = gaußian.copy()
		gaußian_with_spot[0,0] = 10*np.max(gaußian_with_spot)

		excentricity_with_spot = excentricity(gaußian_with_spot,centre=centre,npoints=10)
		assert excentricity_with_spot>original_excentricity

	def test_distortion_increases_excentricity(self):
		eccentricities = []

		for stretch_factor in 10**np.linspace(0,1,10):
			distorted_distances = np.hypot(x,stretch_factor*y)
			distorted_gaussian = gauß_function(distorted_distances)
			eccentricities.append(excentricity(distorted_gaussian,centre=centre,npoints=10))

		assert max(eccentricities[1:]) > eccentricities[0]

	def test_centre_offset_increases_excentricity(self):
		eccentricities = [excentricity(gaußian, centre=centre + [offset, 0], npoints=10)for offset in np.linspace(0, 1, 10)]
		assert_increasing(eccentricities)

	def test_speckle_mask(self):
		data = np.ones((20, 20))
		data[10, 10] = 100

		mask = np.zeros((20, 20), dtype=bool)
		mask[10, 10] = True
		result = excentricity(data,centre=(10, 10),speckle_mask=mask,npoints=20)

		assert result < 1e-10 # a masked bright pixel should not affect excentricity


class TestCircleMask(object):

	def test_output_shape(self):
		mask = circle_mask((100, 80), centre=(50, 40), width=10)

		assert mask.shape == (100, 80)
		assert mask.dtype == bool
		assert mask[50, 40]

	def test_radius(self):
		mask = circle_mask((100, 80), centre=(50, 40), width=10)

		assert mask[50, 45]
		assert mask[55, 40]
		assert not mask[50, 51]
		assert not mask[61, 40]

	def test_zero_radius_includes_only_centre(self):
		mask = circle_mask((100, 80), centre=(50, 40), width=0)

		assert np.sum(mask) == 1
		assert mask[50, 40]