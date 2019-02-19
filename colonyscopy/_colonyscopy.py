import numpy as np
from scipy.signal import argrelmax, argrelmin
from colonyscopy.tools import smoothen, color_distance, expand_mask, color_sum
import matplotlib.pyplot as plt
from warnings import warn

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
		bg_intensity = np.empty((self.n_times,self.n_colours))
		col_intensity = np.empty((self.n_times,self.n_colours))
		intensity = np.empty((self.n_times,self.n_colours))
		for t in range(self.n_times):
			for m in range(self.n_colours):
				bg_intensity[t,m] = np.mean(self.images[t,:,:,m][self.background_mask])
				col_intensity[t,m] = np.mean(self.images[t,:,:,m][self.mask])
		intensity = color_sum((col_intensity - bg_intensity))
		return intensity

	@property
	def mask(self):
		"""
		Returns the mask for colony area for this segment.
		"""
		if not hasattr(self,"_mask"):
			self.create_mask()
		return self._mask

	def create_mask(self, cutoff_factor = 0.5):
		"""
		Creates a mask for colony area in this segment.
		"""
		t = self.threshold_timepoint
		max = np.max(color_sum(self.images[t])[self.speckle_mask])
		min = np.min(color_sum(self.images[t])[self.speckle_mask])
		self._mask = np.empty(self.resolution)
		self._mask = np.multiply(color_sum(self.images[t]),self.speckle_mask) > cutoff_factor * (max+min)

	@property
	def threshold_timepoint(self):
		"""
		Returns the timepoint when intensity thresholding should be done to determine colony area.

		TODO: explain parameter
		"""
		if not hasattr(self,"_threshold_timepoint"):
			self.create_threshold_timepoint()
		return self._threshold_timepoint

	def create_threshold_timepoint(self, seg_intensity_threshold = 1000, smooth_width = 10, growth_threshold = 600):
		a = self.segment_intensity()
		a = smoothen(a, smooth_width)
		try:
			m = list(a > growth_threshold).index(True)
		except(ValueError):
			raise ColonyscopyFailedHeuristic("Growth threshold was not reached. Either growth threshold is chosen too high or there is no growth in this segment.")
		try:
			self._threshold_timepoint = list(a > seg_intensity_threshold).index(True)
		except(ValueError):
			self._threshold_timepoint = -1
			warn("Segment intensity threshold was not reached. Colony area mask was created from last picture in time lapse.")


	@property
	def background_mask(self):
		"""
		Returns the mask for background pixels in this segment.

		TODO: explain parameter
		"""
		if not hasattr(self,"_background_mask"):
			self.create_background_mask()
		return self._background_mask

	def create_background_mask(self, expansion = 4):
		"""
		Creates a mask that only includes background pixels of this segment.

		TODO: explain parameter
		"""
		self._background_mask = np.logical_not(expand_mask(self.mask, width = expansion)) + self.speckle_mask

	def segment_intensity(self):
		seg_intensity = np.array([np.mean(color_sum(self.images[t])[self.speckle_mask]) for t in range(self.n_times)])
		return seg_intensity - np.average(np.sort(seg_intensity)[:7])

	def plot_segment_intensity(self,smooth_width = 10, seg_intensity_threshold = 1000):
		plt.plot(self.segment_intensity(), label='Segment intensity')
		plt.plot(smoothen(self.segment_intensity(), smooth_width), label='Smoothened segment intensity')
		plt.plot(seg_intensity_threshold * np.ones(self.n_times), '--', label='Threshold')
		plt.legend()
		plt.show()

	def display_growth_curve(self):  # New intensity measure
	    N_t = self.n_times
	    time = np.linspace(0,(N_t-1)*0.25,N_t)
	    fit_interval_length = 0.9
	    min_lower_bound = 2.0
	    pl = np.empty((3,N_t))

	    pl = self.colony_intensity()

	    if np.min(pl) < 0:
	        pl = pl+1.05*abs(np.min(pl))

	    smooth_log = smoothen(np.log10(pl)[np.logical_not(np.isnan(np.log10(pl)))], 10)
	    smooth_time = time[np.logical_not(np.isnan(np.log10(pl)))]
	    n_nan = np.sum(np.isnan(np.log10(pl)))

	    lower_bound = (np.max(smooth_log)+np.min(smooth_log)-fit_interval_length)/2

	    if lower_bound < min_lower_bound:
	        lower_bound = min_lower_bound

	    upper_bound = lower_bound + fit_interval_length

	    for k in range(len(smooth_log)):
	        if smooth_log[k] > lower_bound:
	            i_0 = k+n_nan
	            break


	    for k in range(len(smooth_log)):
	        if smooth_log[k] > upper_bound:
	            i_f = k+n_nan
	            break

	    a = np.polyfit(time[i_0:i_f], np.log10(pl[i_0:i_f]), 1)

	    gen_time = np.log10(2)/a[0]

	    print('Calculated generation time in hours is')
	    print(gen_time)

	    plt.figure(figsize=(12,8))
	    plt.plot(time, np.log10(pl), '.', label='Measurement')
	    plt.plot(time[i_0:i_f], np.log10(pl[i_0:i_f]), '.', label='Timepoints included in fit')
	    plt.plot(time[i_0:i_f], a[0]*time[i_0:i_f] + a[1], label='Fit')
	    plt.xlabel('Time [h]')
	    plt.ylabel('Intensity')
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

	def __init__(self,images,layout=(8,12),bg=None):
		self.images = images
		self.layout = np.asarray(layout,dtype=int)
		self.resolution = images.shape[1:3]
		self.n_colours = images.shape[3]
		self.n_times = images.shape[0]
		if bg is None:
			self.background = np.average(images[0],axis=(0,1))
		else:
			self.background = np.asarray(bg)

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
			raise ColonyscopyFailedHeuristic("Could not detect colony coordinates. Check if layout in right order.")

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
					((self.coordinates[i][1:]+self.coordinates[i][:-1])/2).astype(int),
					self.resolution[i]))
				for i in (0,1)
			]

	def _gradient_mask(self,threshold = 1000):
		"""
		Returns pixels in the background with a high gradient.
		"""
		return np.linalg.norm(np.gradient(color_sum(self.background),axis=(0,1)),axis=0) > threshold

	def _intensity_mask(self,factor = 4):
		"""
		Returns pixels of the background whose intensity is outside of `factor` times the standard deviation of all pixels.
		"""
		background_intensity = color_sum(self.background)
		return background_intensity > np.mean(background_intensity)+factor*np.std(background_intensity)

		#TODO: include particularly dark pixels, work with percentiles.

	def _temporal_mask(self,threshold = 1200):
		"""
		Returns pixels where the colour changes suddenly in time.
		"""
		matrix = np.zeros(self.resolution,dtype=bool)
		matrix |= color_distance(self.images[0],self.background) > threshold
		for t in range(self.n_times-1):
			matrix |= color_distance(self.images[t+1],self.images[t]) > threshold
		return matrix

	def create_speckle_mask(self,
				gradient_threshold = 1000,
				intensity_factor = 4,
				temporal_threshold = 1200,
				expansion = 3,
			):
		"""
		Tries to automatically detect speckles.

		TODO:
		explain parameters
		"""
		self._speckle_mask = np.logical_not(expand_mask(
				  self._gradient_mask(gradient_threshold)
				| self._intensity_mask(intensity_factor)
				| self._temporal_mask(temporal_threshold),
				width = expansion
			))

	@property
	def speckle_mask(self):
		if not hasattr(self,"_speckle_mask"):
			self.create_speckle_mask()
		return self._speckle_mask

	def create_colonies(self):
		x = self.borders[0]
		y = self.borders[1]
		self._colonies = np.array(
				[[Colony(self.images[:,x[i]:x[i+1],y[j]:y[j+1],:],
						self.background[x[i]:x[i+1],y[j]:y[j+1],:],
						self.speckle_mask[x[i]:x[i+1],y[j]:y[j+1]]) for i in range(self.layout[0])]
				for j in range(self.layout[1])]
			, dtype=object)

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
