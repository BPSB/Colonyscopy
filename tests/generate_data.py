import numpy as np
from itertools import product

def midpoints(number,length):
	"""
	splits `length` into `number` equidistant intervals and returns an array containing the midpoints of those intervals.
	"""
	return np.linspace(0,length,2*number+1)[1::2]

def colony_centres(layout,resolution):
	"""
	like midpoints, in two dimensions: splits an an area of size `resolution` into a checkerboard with `layout[0]`×`layout[1]` elements.
	Returns a 3D array, containing the coordinates of the centres of each tile.
	"""
	return np.dstack(np.meshgrid(*[
			midpoints(layout[i],resolution[i]) for i in (0,1)
		],indexing="ij"))

def generate_data(colony_sizes,resolution,bg=(0,0,0),fg=(255,255,255)):
	"""
	Genererates artefical datasets for testing purposes.
	
	Parameters
	----------
	colony_sizes : NumPy array
		Radii of the colonies. The dimensions are time×x×y.
	
	resolution : tuple of integers
		The resolution of the resulting image
	
	bg, fg: tuples of three integer
		The colours for the background and colonies, respectively.
	"""
	
	bg = np.asarray(bg,dtype=np.uint8)
	fg = np.asarray(fg,dtype=np.uint8)
	
	times = colony_sizes.shape[0]
	layout = colony_sizes.shape[1:3]
	positions = colony_centres(layout,resolution)
	coordinates = np.indices(resolution).transpose(1,2,0)
	
	result = np.tile(bg,(times,*resolution,1))
	# This loop could be made more concise with NumPy indexing, but it needs less memory.
	for i in range(layout[0]):
		for j in range(layout[1]):
			distances = np.linalg.norm( coordinates-positions[i,j], axis=2 )
			mask = distances<colony_sizes[:,None,None,i,j]
			result[mask] = fg
	
	return result

def test_generate_data():
	times = 4
	layout = (5,7)
	colony_sizes = np.random.uniform(3,10,(times,*layout))
	resolution = (200,100)
	bg = np.random.randint(0,256,3)
	fg = np.random.randint(0,256,3)
	
	data = generate_data(colony_sizes,resolution,bg,fg)
	
	centres = colony_centres(layout,resolution)
	for _ in range(1000):
		time = np.random.randint(times)
		pos = [np.random.randint(resolution[i]) for i in (0,1)]
		is_colony = any(
				np.linalg.norm(pos-centres[i,j]) < colony_sizes[time,i,j]
				for i,j in product(*map(range,layout))
			)
		assert np.all(data[(time,*pos)] == (fg if is_colony else bg))

