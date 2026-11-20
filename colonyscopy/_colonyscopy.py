import numpy as np
from scipy.signal import argrelmax, find_peaks, peak_prominences
from .tools import smoothen, color_distance, expand_mask, color_sum, circle_mask, radial_profile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Rectangle
import ipywidgets as widgets
from IPython.display import display

class ColonyscopyFailedHeuristic(Exception):
	"""
	Raised when an automatic analysis heuristic fails to produce a reliable result.
	Manual adjustment or inspection of the input images may be necessary.
	"""
	pass


class MaskComputationFailed(ColonyscopyFailedHeuristic):
	"""
	Raised when a valid colony or background mask cannot be computed.
	"""
	pass

class Colony(object):
	"""
		Class representing a single colony.

		Parameters
		----------
		images : numpy.ndarray
			A four-dimensional array containing the colony image time series, with the shape (time, height, width, colour channels).

		background : numpy.ndarray
			A three-dimensional array containing the background, with the shape (height, width, colour channels). Typically derived from a reference image of the plate before any colony growth.

		speckle_mask : boolean numpy.ndarray
			Two-dimensional mask with shape (height, width). False indicates pixels excluded due to unwanted image features.

		centre : array-like of int
			Colony centre coordinates (row, column), relative to the colony image segment.


		Attributes
		----------
		resolution : tuple of int
			Spatial image dimensions (height, width).

		n_colours : int
			Number of colour channels.

		n_times : int
			Number of time points in the image series.

		Notes
		-----
		Several calculated properties are cached and are not automatically updated if the underlying image data changes.
	"""

	def __init__(self,images,background,speckle_mask,centre):
		#parameters
		self.images = images
		self.background = background
		self.speckle_mask = speckle_mask
		self.centre = centre

		#attributes
		self.resolution = images.shape[1:3]
		self.n_colours = images.shape[3]
		self.n_times = images.shape[0]

	@property
	def temp_mean(self):
		"""
			Calculates the pixel-wise average across all time points of the image series.

			Returns
			-------
			numpy.ndarray
				Mean image with shape (height, width, channels).
		"""
		if not hasattr(self,"_temp_mean"):
			self._temp_mean = np.average(self.images,axis=0)
		return self._temp_mean

	@property
	def colony_intensity(self):
		"""
			The colony intensities calculated with :meth:`create_colony_intensity`.

			Returns
			-------
			numpy.ndarray
				One-dimensional array of background-corrected colony intensities.
		"""
		if not hasattr(self, "_colony_intensity"):
			self.create_colony_intensity()
		return self._colony_intensity

	def create_colony_intensity(self):
		"""
			Calculates the background-corrected colony intensity over time.

			For each time point, the median background intensity is subtracted from the mean colony intensity.
			If the mean corrected intensity of the first six time points is negative, unusually bright background regions are excluded and the
			intensities are recalculated.

			If background mask computation fails, the intensity time
			series is set to zero.

			Background mask refinement uses recursion. If the recursion limit is reached, intensities are recalculated using the current masks.
		"""
		bg_intensity = np.empty(self.n_times)
		col_intensity = np.empty(self.n_times)
		# Loop is necessary due to mask destroying structure
		try:
			if not np.any(self.mask):
				raise MaskComputationFailed("Colony mask contains no valid pixels.")

			if not np.any(self.background_mask):
				raise MaskComputationFailed("Background mask contains no valid pixels.")

			for t in range(self.n_times):
				col_intensity[t] = np.mean(color_sum(self.images[t,:,:])[self.mask])
				bg_intensity[t] = np.median(color_sum(self.images[t,:,:])[self.background_mask])

			self._colony_intensity = (col_intensity - bg_intensity)

			#negative intensities may be because of bright background artifacts
			if (np.mean(self._colony_intensity[:6]) < 0):
				factor = 1.5
				background_px = color_sum(self.images[0])[self.background_mask]

				bright_px = np.multiply(color_sum(self.images[0]), self.background_mask) > np.mean(background_px)+factor*np.std(background_px)
				bright_px = expand_mask(bright_px, width=3)

				old_mask = self._background_mask.copy()
				self._background_mask &= np.logical_not(bright_px)

				if not np.array_equal(old_mask, self._background_mask):
					try:
						self.create_colony_intensity() #recalculate intensities using the corrected background mask.

					except(RecursionError):
						bg_intensity = np.empty(self.n_times)
						col_intensity = np.empty(self.n_times)
						# Loop is necessary due to mask destroying structure
						for t in range(self.n_times):
							col_intensity[t] = np.mean(color_sum(self.images[t,:,:])[self.mask])
							bg_intensity[t] = np.median(color_sum(self.images[t,:,:])[self.background_mask])
						self._colony_intensity = (col_intensity - bg_intensity)

		except(MaskComputationFailed):
			self._colony_intensity = np.zeros(self.n_times)

	@property
	def mask(self):
		"""
			The mask for colony area for this segment.

			Returns
			-------
			boolean numpy.ndarray
				Two-dimensional mask where True indicates pixels included in colony measurements.
		"""
		if not hasattr(self,"_mask"):
			self.create_mask()
		return self._mask

	def create_mask(self, colony_mask_width = 10):
		"""
			Creates a circular mask for measuring colony intensity in this segment.
			A circular region centred on the colony is combined with the speckle mask. True indicates valid pixels within the circular region.

			Parameters
			----------
			colony_mask_width : int, optional
				Radius of the circular colony measurement region in pixels. Default is 10.

			Notes
			-----
			Calling this method again overwrites the existing colony mask.
		"""
		self._mask = circle_mask(self.resolution, self.centre, colony_mask_width) & self.speckle_mask

	def segment_intensity(self):
		"""
			Calculates the baseline-corrected intensity of the entire segment.
			The baseline is defined as the average of the seven lowest segment intensities across the time series.

			Returns
			-------
			numpy.ndarray
				One-dimensional array containing the baseline-corrected segment intensity for each time point.

			Notes
			-----
			Unlike ``colony_intensity``, this method uses all valid pixels in the segment rather than a circular colony mask.
		"""
		if not np.any(self.speckle_mask):
			raise MaskComputationFailed("No valid pixels remain for segment intensity calculation.")

		seg_intensity = np.array([np.mean(color_sum(self.images[t])[self.speckle_mask]) for t in range(self.n_times)])
		return seg_intensity - np.average(np.sort(seg_intensity)[:7])

	@property
	def background_mask(self):
		"""
			The mask for background pixels in this segment.

			Returns
			-------
			boolean numpy.ndarray
				Two-dimensional mask, where True indicates pixels included in background measurements.
		"""
		if not hasattr(self,"_background_mask"):
			self.create_background_mask()
		return self._background_mask

	@background_mask.setter
	def background_mask(self,value):
		"""
			Sets the background mask if its shape and data type are valid.

			Parameters
			----------
			value : boolean numpy.ndarray
				New background mask with the same shape as ``resolution``.

			Notes
			-----
			If the supplied mask has an incorrect shape or data type, the assignment is ignored without raising an exception.
		"""
		if (np.shape(value) == self.resolution) & (value.dtype == bool):
			self._background_mask = value


	def create_background_mask(self, bg_mask_width = 18):
		"""
			Creates a mask that only includes background pixels of this segment.

			Parameters
			----------
			bg_mask_width : int, optional
				Radius of the circular region excluded from background measurements, in pixels. Default is 18.

			Notes
			-----
			Calling this method again overwrites the existing background mask.
		"""
		self._background_mask = np.logical_not(circle_mask(self.resolution, self.centre, bg_mask_width)) & self.speckle_mask

	@property
	def colony_radial_profile(self):
		"""
			The radial intensity profile of the colony.

			Returns
			-------
			tuple of numpy.ndarray
				Two one-dimensional arrays containing radial distances in pixels and the corresponding estimated mean colour-summed intensities.
		"""
		if not hasattr(self, "_colony_radial_profile"):
			self.create_radial_profile()
		return self._colony_radial_profile

	def create_radial_profile(self,smooth_width=0.5,npoints=60):
		"""
			Calculates the radial intensity profile of the colony using the colour-summed intensity of the final image in the time series.

			Parameters
			----------
			smooth_width : float, optional
				Width of the Gaussian weighting function used for radial smoothing, in pixels. Default is 0.5.

			npoints : int, optional
				Number of evenly spaced radial sampling points between zero and the maximum unmasked distance from the colony centre. Default is 60.

			Notes
			-----
			The radial profile is not automatically recalculated if the underlying image data or speckle mask changes.
		"""
		self._colony_radial_profile = radial_profile(color_sum(self.images[-1]),self.centre,np.logical_not(self.speckle_mask),smooth_width,npoints)


class Plate(object):
	"""
		Class representing a plate containing colonies arranged in a rectangular grid.

		Parameters
		----------
		images : numpy.ndarray
			Four-dimensional image time series with shape (time, height, width, colour channels).

		layout : array-like of int, optional
			Number of colony rows and columns on the plate. Default is (8, 12).

		bg : array-like or None, optional
			Reference background used for colony detection. Default is None.
			If None, a spatially uniform background is estimated from the mean colour values of the first image. Alternatively, a colour vector or background image can be provided.

		Attributes
		----------
		resolution : tuple of int
			Spatial image dimensions (height, width).

		n_colours : int
			Number of colour channels.

		n_times : int
			Number of time points in the image series.

		background : numpy.ndarray
			Reference background used for colony detection.

		Notes
		-----
		Colony detection assumes a regular rectangular grid.
		Several calculated properties are cached and are not automatically updated if the underlying image data changes.
	"""

	def __init__(self,images,layout=(8,12),bg=None):
		#parameters
		self.images = images
		self.layout = np.asarray(layout,dtype=int)
		# attributes
		self.resolution = images.shape[1:3]
		self.n_colours = images.shape[3]
		self.n_times = images.shape[0]

		if bg is None:
			mean_color = np.average(images[0], axis=(0, 1))
			self.background = np.broadcast_to(mean_color, images.shape[1:])

		else:
			self.background = np.asarray(bg)

	@property
	def temp_mean(self):
		"""
			Calculates the pixel-wise average across all time points.

			Returns
			-------
			numpy.ndarray
				Mean image with shape (height, width, colour channels).
		"""
		if not hasattr(self,"_temp_mean"):
			self._temp_mean = np.average(self.images,axis=0)
		return self._temp_mean

	def segment(self):
		"""
			Detects colony positions in the plate image.

			Calculates the colour distance between the temporal mean image and the background.
			Row and column intensity profiles are used to identify colony positions.

			Different smoothing widths are tested until the expected number of intensity peaks is detected in both directions.

			Notes
			-----
			Assumes colonies are arranged in a regular rectangular grid, with one detectable peak per row and column.
			Raises ColonyscopyFailedHeuristic if no suitable smoothing width produces the expected number of peaks.
		"""

		intensities = color_distance(self.temp_mean,self.background)
		profiles = [np.average(intensities,axis=i) for i in (1,0)]

		for smooth_width in range(1,int(self.resolution[0]/self.layout[0])):
			self._coordinates = [None,None]
			counts = []

			for i in (0,1):
				smooth_profile = smoothen(profiles[i],smooth_width)
				self._coordinates[i] = argrelmax(smooth_profile)[0]
				counts.append(len(self._coordinates[i]))


			if counts[0] != self.layout[0] or counts[1] != self.layout[1]:
				continue

			else:
				print(f'Smooth width = {smooth_width}')
				break

		else:
			raise ColonyscopyFailedHeuristic(f"Could not detect colony coordinates. "
											 f"Rows: {counts[0]} peaks detected, "
											 f"expected {self.layout[0]}. "
											 f"Columns: {counts[1]} peaks detected, "
											 f"expected {self.layout[1]}. "
											 f"Check the crop and layout.")

	@property
	def colony_centres(self):
		"""
			The colony centre coordinates relative to each segment.

			Returns
			-------
			numpy.ndarray
				Array with shape (rows, columns, 2), containing the centre coordinates in pixels relative to each colony segment.
		"""
		if not hasattr(self, "_colony_centres"):
			self._colony_centres = np.zeros((self.layout[0],self.layout[1],2), dtype=int)

			for i in range(self.layout[0]):
				for j in range(self.layout[1]):
					self._colony_centres[i][j][0] = int(self.centres[i][j][0] - self.borders[0][i])
					self._colony_centres[i][j][1] = int(self.centres[i][j][1] - self.borders[1][j])

		return self._colony_centres

	@property
	def coordinates(self):
		"""
			The detected colony positions.

			Returns
			-------
			list of numpy.ndarray
				Two one-dimensional arrays containing the row and column positions of the detected colony centres.
		"""
		if not hasattr(self,"_coordinates"):
			self.segment()

		return self._coordinates

	@property
	def centres(self):
		"""
			The colony centre coordinates in the plate image.

			Returns
			-------
			numpy.ndarray
				Array with shape (rows, columns, 2), containing the centre coordinates in pixels relative to the full plate image.
		"""
		return np.dstack(np.meshgrid(*self.coordinates,indexing="ij"))

	@property
	def borders(self):
		"""
			Calculates the boundaries separating individual colony segments.

			Returns
			-------
			list of numpy.ndarray
				Two one-dimensional arrays containing the row and column boundaries of the colony segments.

			Notes
			-----
			Boundaries are placed halfway between neighbouring colony centres, including the image edges.
		"""
		return [np.hstack((0, ((self.coordinates[i][1:]+self.coordinates[i][:-1])/2).astype(int), self.resolution[i])) for i in (0,1)]

	def _gradient_mask(self,factor=10):
		"""
			Detects pixels with unusually high background gradients.

			Parameters
			----------
			factor : float, optional
				Multiplier applied to the scaled median absolute deviation. Default is 10.

			Returns
			-------
			boolean numpy.ndarray
				Two-dimensional mask where True indicates pixels exceeding the gradient threshold.
		"""

		gradients = np.linalg.norm(np.gradient(color_sum(self.background),axis=(0,1)),axis=0)
		median = np.median(gradients)
		mad = np.median(np.abs(gradients - median))
		threshold = median + factor * 1.4826 * mad

		return gradients > threshold

	def _intensity_mask(self,factor = 4):
		"""
			Detects unusually bright pixels in the background image.

			Parameters
			----------
			factor : float, optional
				Number of standard deviations above the mean background intensity used as the threshold. Default is 4.

			Returns
			-------
			boolean numpy.ndarray
				Two-dimensional mask where True indicates pixels exceeding the intensity threshold.
		"""

		background_intensity = color_sum(self.background)
		return background_intensity > np.mean(background_intensity)+factor*np.std(background_intensity)

	def _colony_reference_mask(self, radius=10):
		"""
			Creates a reference mask around colony centres to define regions used to calibrate the temporal change threshold.

			Parameters
			----------
			radius : int, optional
				Radius of the circular regions around colony centres in pixels. Default is 10.

			Returns
			-------
			boolean numpy.ndarray
				Two-dimensional mask where True indicates pixels within the colony reference regions.
		"""

		mask = np.zeros(self.resolution, dtype=bool)
		for row, col in self.centres.reshape(-1, 2):
			mask[int(row), int(col)] = True

		return expand_mask(mask, radius)

	def _temporal_mask(self,radius=10,safety_factor=1.25,percentile=None):
		"""
		Detects unusually large colour changes between consecutive images.

		Parameters
		----------
		radius : int, optional
			Radius of colony reference regions in pixels. Default is 10.

		safety_factor : float, optional
			Multiplier applied to the reference colour-change threshold. Default is 1.25.

		percentile : float, optional
			Percentile of colour changes within colony reference regions. Default is None.
			If 0, the percentile is determined automatically based on the number of images.

		Returns
		-------
		boolean numpy.ndarray
			Two-dimensional mask where True indicates pixels exceeding the temporal change threshold.
		"""
		if percentile is None:
			percentile=100-10**(-3-(self.n_times-10)/40)

		colony_regions=self._colony_reference_mask(radius)
		histogram=np.zeros(766,dtype=np.int64)

		for t in range(self.n_times-1):
			before=self.images[t][colony_regions].astype(np.int16)
			after=self.images[t+1][colony_regions].astype(np.int16)
			difference=np.abs(after-before).sum(axis=-1)
			histogram+=np.bincount(difference,minlength=766)

		cumulative=np.cumsum(histogram)
		rank=(percentile/100)*(cumulative[-1]-1)
		lower_rank=int(np.floor(rank))
		upper_rank=int(np.ceil(rank))
		lower=np.searchsorted(cumulative,lower_rank+1)
		upper=np.searchsorted(cumulative,upper_rank+1)

		reference_change=lower+(rank-lower_rank)*(upper-lower)
		threshold=safety_factor*reference_change

		mask=np.zeros(self.resolution,dtype=bool)

		for t in range(self.n_times-1):
			difference=np.zeros(self.resolution,dtype=np.float64)

			for channel in range(self.n_colours):
				before=self.images[t,:,:,channel].astype(np.int16)
				after=self.images[t+1,:,:,channel].astype(np.int16)
				difference+=np.abs(after-before)

			mask|=difference>threshold

		return mask

	def create_speckle_mask(self,expansion=3,gradient_factor=10,intensity_factor=4,temporal_radius=10,temporal_safety_factor=1.25,temporal_percentile=None,use_gradient=True,use_intensity=True,use_temporal=True):
		"""
			Creates the combined speckle mask for the plate, using selected detection methods. Default are gradient, intensity and temporal masks.
			Detected regions are expanded and excluded from further analysis.
			The resulting mask contains True for valid pixels and False for excluded pixels.

			Parameters
			----------
			expansion : int, optional
				Radius used to expand detected speckles. Default is 3

			gradient_factor : float, optional
				Factor controlling the adaptive gradient threshold. Default is 10.

			intensity_factor : float, optional
				Number of standard deviations above the mean intensity. Default is 4.

			temporal_radius : int, optional
				Radius of colony reference regions. Default is 10.

			temporal_safety_factor : float, optional
				Multiplier applied to the temporal reference threshold. Default is 1.25.

			temporal_percentile : float, optional
				Percentile used for temporal threshold calibration. If None, the percentile is determined automatically based on the number of images. Default is None.

			use_gradient : bool, optional
				Whether to include gradient-based detection.

			use_intensity : bool, optional
				Whether to include intensity-based detection.

			use_temporal : bool, optional
				Whether to include temporal detection.
		"""

		mask = np.zeros(self.resolution, dtype=bool)

		if use_gradient:
			mask |= self._gradient_mask(gradient_factor)
		if use_intensity:
			mask |= self._intensity_mask(intensity_factor)
		if use_temporal:
			mask |= self._temporal_mask(radius=temporal_radius,safety_factor=temporal_safety_factor,percentile=temporal_percentile)

		if expansion > 0:
			mask = expand_mask(mask, width=expansion)

		self._speckle_mask = ~mask

	def create_colonies(self):
		"""
			Creates individual Colony objects from the plate image series.

			The images, reference background and speckle mask are divided into segments using the detected colony boundaries.
			Each segment is assigned its corresponding colony centre.

			The speckle mask must be created using :meth:`create_speckle_mask` before calling this method.
		"""
		x = self.borders[0]
		y = self.borders[1]

		self._colonies = np.array(
				[[Colony(self.images[:,x[i]:x[i+1],y[j]:y[j+1],:],
						self.background[x[i]:x[i+1],y[j]:y[j+1],:],
						self._speckle_mask[x[i]:x[i+1],y[j]:y[j+1]],
						self.colony_centres[i][j])
				  for j in range(self.layout[1])]
				for i in range(self.layout[0])]
			, dtype=object)

	@property
	def colonies(self):
		"""
			The Colony objects belonging to the plate.

			Returns
			-------
			numpy.ndarray
				Two-dimensional object array containing one Colony object per plate position.
		"""
		if not hasattr(self,"_colonies"):
			self.create_colonies()
		return self._colonies

	def all_colonies(self): #generator
		"""
			Iterates over all colonies on the plate.

			Yields
			------
			Colony
				The next Colony object, starting from the first row.
		"""
		for i in range(self.layout[0]):
			for j in range(self.layout[1]):
				yield self.colonies[i][j]

	def __iter__(self):
		"""
			Allows iteration over the rows of colonies.

			Returns
			-------
			iterator
				Iterator that returns one row of Colony objects at a time.
		"""
		return self.colonies.__iter__()

	def __getitem__(self,arg):
		"""
			Accesses colonies using array indexing.

			Parameters
			----------
			arg : int, slice or tuple
				Index used to select colonies from the colony array.

			Returns
			-------
			Colony or numpy.ndarray
				Selected colony or subset of the colony array.
		"""
		return self.colonies[arg]


class Image_cropping:
	"""
		Detects the colony grid using Fourier analysis and crops image time series.

		Parameters
		----------
		plate : Plate
			Plate object providing the mean image, colony layout and background.

		Attributes
		----------
		fourier_results : dict or None
			Detected colony positions, spacing and alignment information.

		fourier_quality : dict or None
			Results of the automatic grid quality assessment.

		cropped_image : numpy.ndarray or None
			Cropped image time series.

		crop_bounds : tuple of int or None
			Crop boundaries (y_start, y_end, x_start, x_end).

		crop_approved : bool
			Whether the proposed crop has been accepted.
	"""

	def __init__(self, plate):
		#parameters
		self.plate = plate
		self.layout = plate.layout
		#attributes
		self.fourier_results = None
		self.fourier_quality = None
		self.cropped_image = None
		self.crop_bounds = None
		self.crop_approved = False

	def auto_crop_fourier(self, plot=True):
		"""
			Automatically detects and evaluates the Fourier colony cropping.
			Requests manual confirmation if the quality checks fail.

			Parameters
			----------
			plot : bool, optional
				Whether to display Fourier detection plots. Default is True.

			Returns
			-------
			numpy.ndarray or None
				Cropped time series if accepted, otherwise None.
		"""
		self.cropped_image = None
		self.crop_bounds = None
		self.crop_approved = False

		self.detect_grid_fourier(plot=plot)
		quality = self._evaluate_fourier_quality()

		for name in ["Rows", "Columns"]:
			print(f"{name}: {quality[name]['matched']}/{quality[name]['expected']} peaks matched")

		if quality["all_passed"]:
			cropped = self.crop_fourier()
			self.crop_approved = True
			return cropped

		else:
			print("Automatic Fourier quality assessment failed.")
			print("Manual inspection required.")
			return self.manual_fourier_quality()

	def detect_grid_fourier(self, min_spacing_factor=0.75, max_spacing_factor=1.25,plot=False):
		"""
			Detects the colony grid using Fourier analysis.

			Uses the Fourier transform to estimate the spacing between colonies.
			The grid positions are then optimized based on the intensity profile and refined by fitting a line to nearby detected peaks.

			Parameters
			----------
			min_spacing_factor : float, optional
				Lower bound for colony spacing relative to expected spacing.

			max_spacing_factor : float, optional
				Upper bound for colony spacing relative to expected spacing.

			plot : bool, optional
				Whether to plot intensity profiles, colony positions,
				and proposed crop boundaries.

			Returns
			-------
			dict
				Fourier detection for rows and columns.
		"""
		intensities = color_distance(self.plate.temp_mean,self.plate.background)
		profiles = [np.mean(intensities,axis=1),np.mean(intensities,axis=0)]
		results = {}

		for name, profile, n_colonies in zip(["Rows", "Columns"],profiles,self.layout):
			n = len(profile)
			signal = profile-np.mean(profile)
			fft = np.abs(np.fft.rfft(signal))
			frequencies = np.fft.rfftfreq(n)

			expected_spacing = n/n_colonies
			min_spacing = expected_spacing*min_spacing_factor
			max_spacing = expected_spacing*max_spacing_factor
			valid = ((frequencies>=1/max_spacing)&(frequencies<=1/min_spacing))

			candidate_indices = np.flatnonzero(valid)
			candidate_peaks, _ = find_peaks(fft[valid])

			if len(candidate_peaks) == 0:
				raise ColonyscopyFailedHeuristic(f"Could not determine colony spacing along {name.lower()}: no Fourier peaks detected.")

			best = candidate_indices[candidate_peaks[np.argmax(fft[valid][candidate_peaks])]]
			spacing = 1/frequencies[best]

			smoothed = gaussian_filter1d(profile,sigma=3)
			offsets = np.arange(n_colonies)*spacing
			best_score = -np.inf
			best_centres = None
			max_start = int(np.floor(n-1-offsets[-1]))

			for start in range(max_start+1):
				centres = np.rint(start+offsets).astype(int)
				score = np.mean(smoothed[centres])

				if score > best_score:
					best_score = score
					best_centres = centres

			peaks, _ = find_peaks(smoothed)
			matched = np.full(n_colonies, np.nan)
			search_radius = spacing/1.5

			for i, centre in enumerate(best_centres):
				nearby = peaks[np.abs(peaks-centre)<=search_radius]
				if len(nearby):
					matched[i] = nearby[np.argmin(np.abs(nearby-centre))]

			valid_matches = np.isfinite(matched)
			indices = np.arange(n_colonies)

			if np.sum(valid_matches) >= 3:
				corrected_spacing, corrected_start = np.polyfit(indices[valid_matches], matched[valid_matches],deg=1)
				corrected_centres = (corrected_start+indices*corrected_spacing)

			else:
				corrected_spacing = spacing
				corrected_centres = best_centres.astype(float)

			final_matched = np.full(n_colonies, np.nan)

			for i, centre in enumerate(corrected_centres):
				nearby = peaks[np.abs(peaks - centre) <= search_radius]

				if len(nearby):final_matched[i] = nearby[np.argmin(np.abs(nearby - centre))]

			if plot:
				crop_start, crop_end = self._get_fourier_bounds(corrected_centres,corrected_spacing,n)

			results[name] = {"spacing": corrected_spacing,"positions": corrected_centres,"matched_count": int(np.sum(np.isfinite(final_matched))),}

			if plot:
				fig, ax = plt.subplots(figsize=(15, 4))
				ax.plot(profile,color="black",alpha=0.6)
				for centre in corrected_centres:
					ax.axvline(centre,color="red",alpha=0.4,linewidth=1)

				ax.axvline(crop_start,color="blue",linestyle="--",linewidth=2)
				ax.axvline(crop_end,color="blue",linestyle="--",linewidth=2)
				ax.annotate(f"Start: {crop_start} px",xy=(crop_start, 1),xycoords=("data", "axes fraction"),xytext=(5, 10),textcoords="offset points", ha="right",va="top",color="blue",fontsize=10)
				ax.annotate(f"End: {crop_end} px",xy=(crop_end, 1),xycoords=("data", "axes fraction"),xytext=(-5, 10),textcoords="offset points",ha="left",va="top",color="blue",fontsize=10)
				ax.set_title(f"{name}")
				ax.set_xlabel("Position (pixels)")
				ax.set_ylabel("Intensity")
				plt.tight_layout()
				plt.show()

		self.fourier_results = results
		return results

	def _evaluate_fourier_quality(self):
		"""
		Checks whether the expected number of colony peaks was reached.

		Returns
		-------
		dict
			Quality assessment for rows and columns, containing the number of matched and expected peaks and a Boolean status for each axis.
			The ``all_passed`` entry indicates whether both axes passed.
		"""
		quality = {}

		for name, expected in zip(["Rows", "Columns"], self.layout):
			matched = self.fourier_results[name]["matched_count"]
			quality[name] = {"matched": matched, "expected": expected, "passed": matched == expected}

		quality["all_passed"] = quality["Rows"]["passed"] and quality["Columns"]["passed"]
		self.fourier_quality = quality

		return quality

	def crop_fourier(self):
		"""
			Crops the plate image series half a colony spacing beyond the first and last detected colony centres.
			Returns
			-------
			numpy.ndarray
				Cropped image time series.

			Raises
			------
			ValueError
				If the Fourier grid has not been detected.
		"""
		timeseries = self.plate.images
		if self.fourier_results is None:
			raise ValueError("Run detect_grid_fourier() before cropping.")

		rows = self.fourier_results["Rows"]["positions"]
		cols = self.fourier_results["Columns"]["positions"]
		row_spacing = self.fourier_results["Rows"]["spacing"]
		col_spacing = self.fourier_results["Columns"]["spacing"]

		y_start, y_end = self._get_fourier_bounds(rows, row_spacing, timeseries.shape[1])
		x_start, x_end = self._get_fourier_bounds(cols, col_spacing, timeseries.shape[2])

		self.crop_bounds = (y_start, y_end, x_start, x_end)
		self.cropped_image = timeseries[:, y_start:y_end, x_start:x_end, :].copy()

		return self.cropped_image

	def _get_fourier_bounds(self, positions, spacing, image_length):
		"""
			Calculates crop boundaries from colony positions and spacing.

			Parameters
			----------
			positions : numpy.ndarray
				Detected colony centre positions along one axis.

			spacing : float
				Distance between neighbouring colony centres in pixels.

			image_length : int
				Image dimension along the corresponding axis.

			Returns
			-------
			tuple of int
				Start and end coordinates of the crop.
		"""

		start = max(0,int(np.floor(positions[0]-spacing/2)))
		end = min(image_length,int(np.ceil(positions[-1]+spacing/2)))
		if start >= end:raise ValueError("Invalid crop boundaries.")

		return start, end

	def manual_fourier_quality(self):
		"""
			Displays the Fourier-derived crop for manual inspection. Allows the user to accept, reject or manually adjust the crop boundaries.

			Returns
			-------
			numpy.ndarray or None
				Cropped image time series if accepted, otherwise None. When manually adjusting the crop, the accepted result is stored in ``self.cropped_image``.
		"""
		timeseries = self.plate.images
		cropped = self.crop_fourier()

		fig, ax = plt.subplots(figsize=(10, 7))
		ax.imshow(cropped[-1])
		ax.set_title("Proposed Fourier crop")
		plt.tight_layout()
		plt.show()

		answer = input("Accept Fourier crop or adjust boundaries? (y/a/n):: ").strip().lower()

		if answer in ("y", "yes"):
			self.crop_approved = True
			return cropped
		elif answer in ("a", "adjust"):
			self.adjust_fourier_crop()
			return None
		else:
			self.cropped_image = None
			self.crop_bounds = None
			self.crop_approved = False
			print("Fourier crop rejected.")
			return None

	def adjust_fourier_crop(self):
		"""
			Allows manual adjustment of Fourier crop boundaries using Jupyter widgets.
			The accepted crop is stored in self.cropped_image, and self.crop_approved is set to True.
		"""
		timeseries = self.plate.images
		original =self.crop_bounds

		top = widgets.BoundedIntText(value=original[0],min=0,max=timeseries.shape[1],description="Top:")
		bottom = widgets.BoundedIntText(value=original[1],min=0,max=timeseries.shape[1],description="Bottom:")
		left = widgets.BoundedIntText(value=original[2],min=0,max=timeseries.shape[2],description="Left:")
		right = widgets.BoundedIntText(value=original[3],min=0,max=timeseries.shape[2],description="Right:")

		fig, ax = plt.subplots(figsize=(12, 8))
		ax.imshow(timeseries[-1])
		rectangle = Rectangle((original[2], original[0]),original[3]-original[2], original[1]-original[0],fill=False,edgecolor="yellow",linewidth=2)
		ax.add_patch(rectangle)
		ax.set_title("Manual Fourier crop adjustment")

		plt.close(fig)
		preview = widgets.Output()
		status = widgets.Output()

		def update_preview(change=None):
			y1, y2, x1, x2 = top.value,bottom.value,left.value,right.value
			rectangle.set_xy((x1,y1))
			rectangle.set_width(x2-x1)
			rectangle.set_height(y2-y1)

			rectangle.set_edgecolor("yellow" if y1<y2 and x1<x2 else "red")

			with preview:
				preview.clear_output(wait=True)
				display(fig)

		for field in (top,bottom,left,right):
			field.observe(update_preview, names="value")

		accept_button = widgets.Button(description="Accept crop",button_style="success")
		reset_button = widgets.Button(description="Reset")
		cancel_button = widgets.Button(description="Cancel",button_style="danger")

		def accept(button):
			y1, y2, x1, x2 = top.value,bottom.value,left.value,right.value

			if y1 >= y2 or x1 >= x2:
				with status:
					status.clear_output()
					print("Invalid crop boundaries.")
				return

			self.crop_bounds = (y1, y2, x1, x2)
			self.cropped_image = timeseries[:, y1:y2, x1:x2, :].copy()
			self.crop_approved = True

			with status:
				status.clear_output()
				print("Crop accepted!")
				print("Final crop boundaries:",self.crop_bounds)
				print("Cropped image shape:",self.cropped_image.shape)

		def reset(button):
			for field, value in zip((top,bottom,left,right),original):
				field.value = value

			with status:
				status.clear_output()

		def cancel(button):
			self.cropped_image = None
			self.crop_bounds = None
			self.crop_approved = False

			with status:
				status.clear_output()
				print("Crop adjustment cancelled.")

		accept_button.on_click(accept)
		reset_button.on_click(reset)
		cancel_button.on_click(cancel)
		self._crop_widgets = (top, bottom, left, right, accept_button, reset_button, cancel_button, preview, status)
		self.crop_approved = False
		self.cropped_image = None

		display(widgets.VBox([widgets.HBox([top, bottom]),widgets.HBox([left, right]),widgets.HBox([accept_button, reset_button, cancel_button]),status,preview]))
		update_preview()
