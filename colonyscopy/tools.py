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
	kernel = blackman(width)
	kernel /= np.sum(kernel)
	return np.convolve(data,kernel,mode="same")

def smoothen_image(image,width):
	kernel = blackman(width)[:,None]*blackman(width)
	kernel /= np.sum(kernel)
	result = np.empty_like(image)
	for i in range(3):
		result[:,:,i] = convolve2d(image[:,:,i],kernel,mode="same")
	return result

def expand_mask(mask,width):
	"""
	Expands `mask` (array of booleans) by setting all elments within a distance of `width` to a True element to True.
	"""
	centre = np.array([width,width])
	coordinates = np.indices((2*width+1,2*width+1))
	distances = np.linalg.norm(coordinates-centre[:,None,None],axis=0)
	kernel = (distances <= width).astype(float)
	return convolve2d(mask,kernel,mode="same").astype(bool)

def color_sum(data):
	return np.sum(data, axis=-1)

def color_distance(data1,data2):
	"""
	Determines the L1 norm between the two input arrays in color space. The color axis must be the last axis.
	"""
	return np.sum(np.abs(np.subtract(data1,data2,dtype=float)),axis=-1)

class TestColorDistance(object):
	def test_symmetry(self):
		size = (200,100,3)
		x = np.random.randint(0,256,size=size,dtype=np.uint8)
		y = np.random.randint(0,256,size=size,dtype=np.uint8)
		np.testing.assert_array_equal(
				color_distance(x,y),
				color_distance(y,x),
			)

	def test_shape(self):
		size = (200,100,3)
		x = np.random.randint(0,256,size=size,dtype=np.uint8)
		y = np.random.randint(0,256,size=size,dtype=np.uint8)
		np.testing.assert_array_equal(color_distance(x,y).shape,size[:-1])

def show_image(array,mode="RGB"):
	img = Image.fromarray(array,mode)
	img.show()
	time.sleep(3)
	for proc in psutil.process_iter():
		if proc.name() == "display":
			proc.kill()

def gaussian_profile(abscissae,position,width):
	profile = np.exp(-((abscissae-position)/width)**2/2)
	return profile/sum(profile)

def radial_profile(data,centre,speckle_mask=None,smooth_width=0.5,npoints=60):
    radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None] ) )
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
    abscissae,values = radial_profile(data,centre,speckle_mask,smooth_width=2.0,npoints=60)
    radii = np.hypot( *( np.indices(data.shape)-np.asarray(centre)[:,None,None] ) )
    radii = ma.masked_array(radii, mask=speckle_mask)
    data = ma.masked_array(data, mask=speckle_mask)

    sumsq = 0
    for i,line in enumerate(data):
        for j,intensity in enumerate(line):
            if not data[i,j] is ma.masked:
                expected_intensity = np.interp(radii[i,j],abscissae,values)
                sumsq += (intensity - expected_intensity)**2
    return np.sqrt(sumsq)/data.size

def circle_mask(resolution, centre, width):
    circle_mask = np.zeros(resolution, dtype=bool)
    circle_mask[tuple(centre)] = True
    circle_mask = expand_mask(circle_mask, width)
    return circle_mask
