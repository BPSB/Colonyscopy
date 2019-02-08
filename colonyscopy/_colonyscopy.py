import numpy as np
from scipy.signal import argrelmax, argrelmin, savgol_filter
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
	
	def gradient_mask(self,threshold=3000):
		"""
		Returns pixels in the background with a high gradient.
		"""
		return np.linalg.norm(np.gradient(self.background,axis=(0,1)),axis=(0,3)) > threshold
	
	def intensity_mask(self,factor):
		"""
		Returns pixels of the background whose intensity is outside of `factor` times the standard deviation of all pixels.
		"""
		background_intensity = np.sum(self.background,axis=-1)
		return background_intensity > np.mean(background_intensity)+factor*np.std(background_intensity)
		
		#TODO: include particularly dark pixels, work with percentiles.

	def create_speckle_mask(self):
		a = np.sum(np.gradient(self.background)[0]+np.gradient(self.background)[1], axis=-1) > 3000
		b = np.sum(self.background,axis=-1) > np.mean(np.sum(self.background,axis=-1))+4*np.std(np.sum(self.background,axis=-1))
		self._speckle_mask = a+b+self.temp_speckle_mask()
		self._speckle_mask = np.logical_not(smoothen_mask(self._speckle_mask,4))

	def temp_speckle_mask(self):
		matrix = np.zeros((np.shape(MyFirstPlate.images)[1],np.shape(MyFirstPlate.images)[2]),dtype=bool)
		matrix[:,:] += (color_distance(self.images[0,:,:,:],self.background[:,:,:]) > 1200)
		for t in range(np.shape(MyFirstPlate.images)[0]-1):
			matrix[:,:] += (color_distance(self.images[t+1,:,:,:],self.images[t,:,:,:]) > 1200)
		return matrix
	
	@property
	def speckle_mask(self):
		"""
		Returns the mask for colony area for this plate.
		"""
		if not hasattr(self,"_speckle_mask"):
			self.create_speckle_mask()
		return self._speckle_mask

