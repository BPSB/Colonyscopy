from scipy.signal.windows import blackman
from scipy.signal import convolve2d
import numpy as np
import numpy.ma as ma
from PIL import Image
from pathlib import Path
import time
import psutil
import rawpy

def load_images(paths):
	"""
		Loads an image time series from a provided order of file paths.
		CR2 files are decoded using rawpy, while other supported image formats are loaded using Pillow.

		Parameters
		----------
		paths : iterable of str or pathlib.Path
			Paths to the images in the desired time series order.

		Returns
		-------
		numpy.ndarray
			Image time series with shape (time, height, width, channels) for RGB images.

		Raises
		------
		ValueError
			If no images are provided, an unsupported image format is encountered, or image shapes or data types do not match.

		RuntimeError
			If a RAW image cannot be decoded.
	"""
	paths = [Path(path) for path in paths]
	extensions = {".cr2",".jpg",".jpeg",".png",".tif",".tiff"}

	if not paths:
		raise ValueError("No image files provided")

	for path in paths:
		if path.suffix.lower() not in extensions:
			raise ValueError(f"Unsupported image format: {path}")

	def read_image(path):
		if path.suffix.lower() == ".cr2":
			with rawpy.imread(str(path)) as raw:
				return raw.postprocess(output_bps=8,use_camera_wb=True)
		else:
			with Image.open(path) as img:
				return np.array(img)

	try:
		first_frame = read_image(paths[0])
	except rawpy.LibRawDataError as e:
		raise RuntimeError(f"Failed to decode image 1/{len(paths)}: {paths[0]}") from e

	images = np.empty((len(paths),*first_frame.shape),dtype=first_frame.dtype)
	images[0] = first_frame
	del first_frame

	print(f"\rLoaded 1/{len(paths)} images",end="",flush=True)

	for i,path in enumerate(paths[1:],start=1):
		try:
			frame = read_image(path)
		except rawpy.LibRawDataError as e:
			raise RuntimeError(f"Failed to decode image {i+1}/{len(paths)}: {path}") from e

		if frame.shape != images.shape[1:] or frame.dtype != images.dtype:
			raise ValueError(f"Image shape or dtype mismatch: {path}")

		images[i] = frame
		del frame
		print(f"\rLoaded {i+1}/{len(paths)} images",end="",flush=True)

	print()
	return images

def smoothen(data,width):
	"""
		Parameters
		----------
		data : numpy.ndarray
			One-dimensional array to smooth.

		width : int
			Length of the smoothing window.

		Returns
		-------
		numpy.ndarray
			One-dimensional array of smoothed values obtained by using a Blackman window and convolution.
	"""
	kernel = blackman(width)
	kernel /= np.sum(kernel)
	return np.convolve(data,kernel,mode="same")

def smoothen_image(image,width):
	"""
		Parameters
		----------
		image : numpy.ndarray
			Image with shape (height, width, 3).

		width : int
			Length of the smoothing window along each axis.

		Returns
		-------
		numpy.ndarray
			Three-dimensional array, with the same shape and dtype as the input, obtained using a two-dimensional Blackman window.
	"""
	kernel = blackman(width)[:,None]*blackman(width)
	kernel /= np.sum(kernel)
	result = np.empty_like(image)
	for i in range(3):
		result[:,:,i] = convolve2d(image[:,:,i],kernel,mode="same")
	return result

def expand_mask(mask,width):
	"""
		Parameters
		----------
		mask : numpy.ndarray
			Two-dimensional boolean mask, where True indicates pixels to expand.

		width : int
			Expansion radius in pixels.

		Returns
		-------
		numpy.ndarray
			Two-dimensional boolean mask with shape (height, width), where True indicates pixels within the expanded regions.
	"""
	centre = np.array([width,width])
	coordinates = np.indices((2*width+1,2*width+1))
	distances = np.linalg.norm(coordinates-centre[:,None,None],axis=0)
	kernel = (distances <= width).astype(float)
	return convolve2d(mask,kernel,mode="same").astype(bool)

def color_sum(data):
	"""
		Parameters
		----------
		data : numpy.ndarray
			Array with colour channels along the last axis.

		Returns
		-------
		numpy.ndarray
			Array of summed intensities with the colour axis removed.
	"""
	return np.sum(data, axis=-1)

def color_distance(data1,data2):
	"""
		Parameters
		----------
		data1, data2 : numpy.ndarray
			Arrays with compatible shapes and with colour channels along the last axis.

		Returns
		-------
		numpy.ndarray
			Sum of absolute colour-channel differences between colour vectors.
	"""
	return np.sum(np.abs(np.subtract(data1,data2,dtype=float)),axis=-1)

def gaussian_profile(abscissae,position,width):
	"""
		Parameters
		----------
		abscissae : numpy.ndarray
			Positions at which the Gaussian is evaluated.

		position : float
			Centre of the Gaussian.

		width : float
			Standard deviation of the Gaussian.

		Returns
		-------
		numpy.ndarray
			Gaussian weights normalised to sum to one.
	"""
	profile = np.exp(-((abscissae-position)/width)**2/2)
	return profile/sum(profile)

def radial_profile(data,centre,speckle_mask=None,smooth_width=0.5,npoints=60):
	"""
		Parameters
		----------
		data : numpy.ndarray
			Two-dimensional intensity image.

		centre : array-like of float
			Centre coordinates (row, column).

		speckle_mask : boolean numpy.ndarray, optional
			Boolean mask where True indicates excluded pixels. Default is None.

		smooth_width : float, optional
			Standard deviation of the Gaussian radial weights, in pixels. Default is 0.5.

		npoints : int, optional
			Number of radial positions. Default is 60.

		Returns
		-------
		abscissae : numpy.ndarray
			Radial distances from the centre.

		values : numpy.ndarray
			One-dimensional array containing Gaussian-weighted mean intensity at each radial distance.
	"""
	radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None]))
	radii = ma.masked_array(radii, mask=speckle_mask)
	data = ma.masked_array(data, mask=speckle_mask)

	r_max = np.max(radii)
	abscissae = np.linspace(0,r_max,npoints)
	sums = np.zeros_like(abscissae)
	normalisation = np.zeros_like(abscissae)

	for i,radii_line in enumerate(radii):
		for j,radius in enumerate(radii_line):
			if not data[i,j] is ma.masked:
				profile = gaussian_profile(abscissae,radius,smooth_width)
				normalisation += profile
				sums += profile*data[i,j]
	return abscissae,sums/normalisation

def excentricity(data,centre,speckle_mask=None,smooth_width=0.5,npoints=60):
	"""
		Parameters
		----------
		data : numpy.ndarray
			Two-dimensional intensity image.

		centre : array-like of float
			Centre coordinates (row, column).

		speckle_mask : boolean numpy.ndarray, optional
			Boolean mask where True indicates excluded pixels. Default is None.

		smooth_width : float, optional
			Standard deviation of the Gaussian radial weights, in pixels. Default is 0.5.

		npoints : int, optional
			Number of radial positions. Default is 60.

		Returns
		-------
		float
			Square root of the summed squared deviations from the radial profile, divided by the number of valid pixels.
	"""
	abscissae,values = radial_profile(data,centre,speckle_mask,smooth_width=smooth_width,npoints=npoints)
	radii = np.hypot(*(np.indices(data.shape)-np.asarray(centre)[:,None,None]))
	radii = ma.masked_array(radii, mask=speckle_mask)
	data = ma.masked_array(data, mask=speckle_mask)

	sumsq = 0
	for i,line in enumerate(data):
		for j,intensity in enumerate(line):
			if not data[i,j] is ma.masked:
				expected_intensity = np.interp(radii[i,j],abscissae,values)
				sumsq += (intensity - expected_intensity)**2
	valid = ~ma.getmaskarray(data)
	return np.sqrt(sumsq)/np.count_nonzero(valid)

def circle_mask(resolution, centre, width):
	"""
		Parameters
		----------
		resolution : tuple of int
			Mask dimensions (height, width).

		centre : array-like of int
			Centre coordinates (row, column).

		width : int
			Radius of the circle in pixels.

		Returns
		-------
		numpy.ndarray
			Two-dimensional boolean mask with the same shape as ``resolution``, where True indicates pixels inside the circle.
	"""
	circle_mask = np.zeros(resolution, dtype=bool)
	circle_mask[tuple(centre)] = True
	circle_mask = expand_mask(circle_mask, width)
	return circle_mask
