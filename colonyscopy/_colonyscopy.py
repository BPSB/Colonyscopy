import numpy as np
from scipy.signal import argrelmax, argrelmin
from colonyscopy.tools import smoothen, color_distance, expand_mask
import matplotlib.pyplot as plt

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
			A three-dimensional array containing the background, i.e., what to expect in the absence of a colony. The first two dimensions are spacial; the last dimension has length 3 and is colour.
	"""

	def __init__(self,images,background,speckle_mask):
		self.images = images
		self.background = background
		self.speckle_mask = speckle_mask
		self.resolution = images.shape[1:3]
		self.n_colours = images.shape[3]
		self.n_times = images.shape[0]

	def colony_intensity(self):
		bg_intensity = np.empty((self.n_colours,self.n_times))
		col_intensity = np.empty((self.n_colours,self.n_times))
		intensity = np.empty((self.n_colours,self.n_times))
		for t in range(self.n_times):
			for m in range(self.n_colours):
				bg_intensity[m,t] = np.sum(np.multiply(self.background_mask,self.images[t,:,:,m]))/np.sum(self.background_mask)
				col_intensity[m,t] = np.sum(np.multiply(self.mask,self.images[t,:,:,m]))/np.sum(self.mask)
		intensity = np.sum(col_intensity - bg_intensity, axis=0)
		return intensity

	def create_mask(self, seg_intensity_threshold = 1000, smooth_width = 5, cutoff_factor = 0.5):
		"""
		Creates a mask for colony area in this segment.
		"""
		self._mask = np.empty(self.resolution)
		a = [np.sum(np.multiply(color_distance(self.images[t],self.background),self.speckle_mask))/(np.sum(self.speckle_mask)) for t in range(self.n_times)]
		print(a)
		a = smoothen(a, smooth_width)
		try:
			t = list(a > seg_intensity_threshold).index(True)
		except(ValueError):
			t = -1
			print("Segment intensity threshold was not reached. Colony area mask was created from last picture in time lapse.")
		self._mask = np.multiply(np.sum(self.images[t], axis=-1), self.speckle_mask) > cutoff_factor * np.max(np.sum(self.images[t], axis=-1))

	@property
	def background_mask(self, expansion = 4):
		"""
		Returns the mask for background pixels in this segment.

		TODO: explain parameter
		"""
		if not hasattr(self,"_background_mask"):
			self.create_background_mask(expansion)
		return self._background_mask

	def create_background_mask(self, expansion = 4):
		"""
		Creates a mask that only includes background pixels of this segment.

		TODO: explain parameter
		"""
		self._background_mask = np.logical_not(expand_mask(self.mask, width = expansion)) + np.logical_not(self.speckle_mask)

	@property
	def mask(self, seg_intensity_threshold = 1000, smooth_width = 5, cutoff_factor = 0.5):
		"""
		Returns the mask for colony area for this segment.
		"""
		if not hasattr(self,"_mask"):
			self.create_mask(seg_intensity_threshold, smooth_width, cutoff_factor)
		return self._mask

	def segment_intensity(self):
		return np.array([np.sum(np.sum(np.multiply(np.sum(self.images[t],axis=-1),self.speckle_mask),axis=-1),axis=-1) for t in range(self.n_times)]/np.sum(self.speckle_mask))

	def plot_segment_intensity(self,smooth_width = 5, seg_intensity_threshold = 1000):
		plt.plot(self.segment_intensity(), label='Segment intensity')
		plt.plot(smoothen(self.segment_intensity(), smooth_width), label='Smoothened segment intensity')
		plt.plot(seg_intensity_threshold * np.ones(self.n_times), '--', label='Threshold')
		plt.legend()
		plt.show()


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
		self.n_colours = images.shape[3]
		self.n_times = images.shape[0]
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
				yield self.colonies[i][j]

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
