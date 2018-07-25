import numpy as np
from scipy.signal import argrelmax
from colonyscopy.tools import smoothen, color_distance

class ColonyscopyFailedHeuristic(Exception):
	pass

class Colony(object):
	"""
		Class representing a single colony.
		
		Parameters
		----------
		images : NumPy array
			A four-dimensional array containing the data for the actual colony. This can be a slice of a bigger array. The first dimension is time; the next two dimensions are spacial; the last dimension has length 3 and is colour.
		
		centre : pair of integers
			The centre of the colony in the coordinates of `images`.
	"""
	
	def __init__(self,images,centre):
		self.images = images
		self.centre = np.asarray(centre,dtype=int)

class Plate(object):
	"""
		The basic class for handling plates with colonies.
		
		Parameters
		----------
		images : NumPy array
			A four-dimensional array containing all the raw data.
			The first dimension is time; the next two dimensions are spacial; the last dimension has length 3 and is colour.
		
		format : pair of integers
			The number of colonies in each direction of the plate.
		
		bg : array-like or None
			The background, i.e., what to expect in absence of colonies.
			
			* If `None`, the background will be guessed automatically.
			* If a length-three sequence, the background will be taken to be homogeneous in that color.
			* If an array with the same dimensions as an image, this will be taken as a background image.
	"""
	
	def __init__(self,images,format=(12,8),bg=None):
		self.images = images
		self.format = np.asarray(format,dtype=int)
		self.resolution = images.shape[1:3]
		if bg is None:
			self.background = np.average(images[0],axis=(0,1))
		else:
			self.background = np.asarray(bg,dtype=np.uint8)
	
	@property
	def temp_mean(self):
		if not hasattr(self,"_temp_mean"):
			self._temp_mean = np.average(self.images,axis=0)
		return self._temp_mean
	
	def segment(self):
		"""
		Tries to automatically segment the images into individual colonies.
		"""
		intensities = color_distance(self.temp_mean,self.background)
		profiles = [np.average(intensities,axis=i) for i in (1,0)]
		
		for smooth_width in range(1,int(self.resolution[0]/self.format[0])):
			smooth_profiles = [smoothen(profiles[i],smooth_width) for i in (0,1)]
			self._coordinates = [argrelmax(smooth_profiles[i])[0] for i in (0,1)]
			if all(
					len(self._coordinates[i]) == self.format[i]
					for i in (0,1)
				):
				break
		else:
			raise ColonyscopyFailedHeuristic("Could not detect colony coordinates. ")
		# for smooth_width in range(1,int(self.resolution[0]/self.format[0])):
		# 	self._coordinates = [None,None]
		# 	for i in (0,1):
		# 		smoothened_profile = smoothen(profiles[i],smooth_width)
		# 		self._coordinates[i] = argrelmax(smoothened_profile)[0]
		# 		if len(self._coordinates[i]) != self.format[i]:
		# 			# Continue outer loop:
		# 			print(self._coordinates)
		# 			break
		# 	else:
		# 		# If inner loop was not broken (i.e., the number of maxima and the format match), break outer loop:
		# 		break
		# else:
		# 	# If outer loop was not broken:
		# 	raise ColonyscopyFailedHeuristic("Could not detect colony coordinates. ")
	
	@property
	def coordinates(self):
		"""
		Returns a pair of one-dimensional arrays containing for each direction the coordinates where colonies are placed.
		"""
		if not hasattr(self,"_coordinates"):
			self.segment()
		return self._coordinates

	@property
	def centres(self):
		"""
		Returns a three dimensional array, containing the centres of the colonies by two-dimensional index.
		"""
		return np.dstack(np.meshgrid(*self.coordinates,indexing="ij"))

	@property
	def borders(self):
		"""
		Returns a pair of one-dimensional arrays containing for each direction the positions of borders of colonies (including 0 and the image size).
		"""
		return [
				np.hstack((
					0,
					(self.coordinates[i][1:]+self.coordinates[i][:-1])/2,
					self.resolution[i]))
				for i in (0,1)
			]


