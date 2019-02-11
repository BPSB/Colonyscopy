import numpy as np
from scipy.signal import argrelmax, argrelmin
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
		
		background : NumPy array
			A three-dimensional array containing the background, i.e., what to expect in the absence of a colony. Thi first two dimensions are spacial; the last dimension has length 3 and is colour.
	"""
	
	def __init__(self,images,background):
		self.images = images
		self.background = background
	
	def intensity(self):
		return sum(
				np.sum(color_distance(image,self.background))
				for image in images
			)

class Plate(object):
	"""
		The basic class for handling plates with colonies.
		
		Parameters
		----------
		images : NumPy array
			A four-dimensional array containing all the raw data.
			The first dimension is time; the next two dimensions are spacial; the last dimension has length 3 and is colour.
		
		layout : pair of integers
			The number of colonies in each direction of the plate.
		
		bg : array-like or None
			The background, i.e., what to expect in absence of colonies.
			
			* If `None`, the background will be guessed automatically.
			* If a length-three sequence, the background will be taken to be homogeneous in that color.
			* If an array with the same dimensions as an image, this will be taken as a background image.
	"""
	
	def __init__(self,images,layout=(12,8),bg=None):
		self.images = images
		self.layout = np.asarray(layout,dtype=int)
		self.resolution = images.shape[1:3]
		self.n_colours = 3
		self.n_times = images.shape[3]
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
		
		for smooth_width in range(1,int(self.resolution[0]/self.layout[0])):
			self._coordinates = [None,None]
			for i in (0,1):
				smooth_profile = smoothen(profiles[i],smooth_width)
				self._coordinates[i] = argrelmax(smooth_profile)[0]
				if len(self._coordinates[i]) != self.layout[i]:
					# bad smooth width → continue outer loop
					break
			else:
				# good smooth width → break outer loop
				break
		else:
			# no good smooth width at all:
			raise ColonyscopyFailedHeuristic("Could not detect colony coordinates.")
	
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
	
	def _gradient_mask(self,threshold):
		"""
		Returns pixels in the background with a high gradient.
		"""
		return np.linalg.norm(np.gradient(self.background,axis=(0,1)),axis=(0,3)) > threshold
	
	def _intensity_mask(self,factor):
		"""
		Returns pixels of the background whose intensity is outside of `factor` times the standard deviation of all pixels.
		"""
		background_intensity = np.sum(self.background,axis=-1)
		return background_intensity > np.mean(background_intensity)+factor*np.std(background_intensity)
		
		#TODO: include particularly dark pixels, work with percentiles.
	
	def _temporal_mask(self,threshold):
		"""
		Returns pixels where the colour changes suddenly in time.
		"""
		matrix = np.zeros(self.resolution,dtype=bool)
		matrix |= color_distance(self.images[0],self.background) > threshold
		for t in range(self.n_times):
			matrix |= color_distance(self.images[t+1],self.images[t]) > threshold
		return matrix
	
	def create_speckle_mask(self,
				gradient_threshold = 3000,
				intensity_factor = 4,
				temporal_threshold = 1200,
				expansion = 4,
			):
		"""
		Tries to automatically detect speckles.
		
		TODO:
		explain parameters
		"""
		self._speckle_mask = expand_mask(
				  self._gradient_mask(gradient_threshold)
				& self._intensity_mask(intensity_factor)
				& self._temporal_mask(temporal_threshold),
				width = expansion
			)
	
	@property
	def speckle_mask(self):
		if not hasattr(self,"_speckle_mask"):
			self.create_speckle_mask()
		return self._speckle_mask
	
	def create_colonies(self):
		self._colonies = np.array([
				[Colony(TODO) for i in layout[0]]
				for j in layout[1]
			],dtype=object)
	
	@property
	def colonies(self):
		if not hasattr(self,"_colonies"):
			self.create_colonies()
		return self._colonies
	
	def all_colonies(self):
		"""
			Returns a generator over all colonies of the plate.
		"""
		for i in layout[0]:
			for j in layout[1]:
				yield self._colonies[i][j]
	
	def __iter__(self):
		return self.colonies.__iter__()
	
	def __getitem__(self,arg):
		return self.colonies[arg]
	
	def localise_colonies(self,*args,**kwargs):
		"""
			Calls colony.localise for all colonies (with the respective arguments).
		"""
		for colony in self.all_colonies():
			colony.localise(*args,**kwargs)




