import numpy as np
from itertools import product
from pytest import approx, mark
from colonyscopy.tools import smoothen_image

def midpoints(number,length):
	"""
	 Splits `length` into `number` equidistant intervals and returns an array containing the midpoints of those intervals.
	"""
	return np.linspace(0,length,2*number+1)[1::2]

def colony_centres(layout,resolution):
	"""
		Splits an an area of size `resolution` into a checkerboard with `layout[0]`×`layout[1]` elements.
		Returns a 3D array, containing the coordinates of the centres of each tile.
	"""
	return np.dstack(np.meshgrid(*[midpoints(layout[i],resolution[i]) for i in (0,1)],indexing="ij"))

def generate_data(colony_sizes, resolution,bg=(0,0,0), fg=(255,255,255),noise_amplitude=0, noise_width=10, rim_width=0, rim_color=(255,255,255)):
	"""
		Genererates artifical datasets for testing purposes.

	Parameters
	----------
	colony_sizes : NumPy array
		Radii of the colonies. The dimensions are time×x×y.

	resolution : tuple of integers
		The resolution of the resulting image

	bg, fg: tuples of three integer
		The colours for the background and colonies, respectively.
		bg can also be an image, which is then used as a background for each generated image.

	rim_width, rim_color:
		Width and color of a rim surrounding each colony.

	noise_amplitude:
		Amplitude of normally distributed noise added to each pixel.
		This does not account for over- and underflows, but the methods have to cope with single bright or dark pixels anyway.
	"""
	bg = np.asarray(bg,dtype=np.uint8)
	fg = np.asarray(fg,dtype=np.uint8)
	rim_color = np.asarray(rim_color,dtype=np.uint8)

	times = colony_sizes.shape[0]
	layout = colony_sizes.shape[1:3]
	positions = colony_centres(layout,resolution)
	coordinates = np.indices(resolution).transpose(1,2,0)

	if bg.shape==(3,):
		result = np.tile(bg,(times,*resolution,1))
	else:
		result = np.tile(bg,(times,1,1,1))

	for i in range(layout[0]):
		for j in range(layout[1]):
			distances = np.linalg.norm( coordinates-positions[i,j], axis=2 )
			mask = distances<colony_sizes[:,None,None,i,j]
			result[mask] = fg

	if rim_width:
		for i in range(layout[0]):
			for j in range(layout[1]):
				distances = np.linalg.norm( coordinates-positions[i,j], axis=2 )
				rim_mask = np.logical_and(colony_sizes[:,None,None,i,j]-rim_width/2<distances, colony_sizes[:,None,None,i,j]+rim_width/2>distances )
				result[rim_mask] = rim_color

	if noise_amplitude:
		for time in range(times):
			noise = np.random.normal(0,noise_amplitude,(*resolution,3))
			if noise_width:
				noise = smoothen_image(noise,noise_width)

			noisy_image = result[time].astype(np.float32)+noise
			result[time] = np.clip(noisy_image, 0, 255).astype(np.uint8)
	return result

cases = mark.parametrize("homogeneous_bg",[False,True])

@cases
def test_generate_data(homogeneous_bg):
	times = 4
	layout = (5,7)
	colony_sizes = np.random.uniform(3,10,(times,*layout))
	resolution = (200,100)

	fg,rim_color = np.random.randint(0,256,(2,3))
	if homogeneous_bg:
		bg = np.random.randint(0,256,3)
	else:
		bg = np.random.randint(0,256,(*resolution,3))
	rim_width = 3
	data = generate_data(colony_sizes,resolution,bg,fg,0,0,rim_width,rim_color)
	centres = colony_centres(layout,resolution)

	for _ in range(1000):
		time = np.random.randint(times)
		pos = [np.random.randint(resolution[i]) for i in (0,1)]
		is_colony = any(np.linalg.norm(pos-centres[i,j]) < colony_sizes[time,i,j] for i,j in product(*map(range,layout)))
		is_rim = any(colony_sizes[time,i,j]-rim_width/2 < np.linalg.norm(pos-centres[i,j]) < colony_sizes[time,i,j]+rim_width/2 for i, j in product(*map(range, layout)) )

		if is_rim:
			assert np.all(data[(time,*pos)] == rim_color)
		elif is_colony:
			assert np.all(data[(time,*pos)] == fg)
		else:
			if homogeneous_bg:
				assert np.all(data[(time,*pos)] == bg)
			else:
				assert np.all(data[(time,*pos)] == bg[tuple(pos)])

@cases
def test_noise(homogeneous_bg):
	times = 4
	layout = (5,7)
	colony_sizes = np.random.uniform(3,10,(times,*layout))
	resolution = (200,100)

	fg,rim_color = np.random.randint(0,256,(2,3))
	if homogeneous_bg:
		bg = np.random.randint(0,256,3)
	else:
		bg = np.random.randint(0,256,(*resolution,3))

	rim_width = 3
	noise_amplitude = 5
	data = [generate_data(colony_sizes,resolution,bg,fg,noise,0,rim_width,rim_color) for noise in (0,noise_amplitude)]

	amplitude_estimate = np.std((data[0].astype(np.int16)-data[1].astype(np.int16)))
	assert amplitude_estimate==approx(noise_amplitude,abs=1)

