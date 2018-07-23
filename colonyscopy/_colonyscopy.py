import numpy as np

class ColonyArray(object):
	"""
		The basic class for handling colony arrays.
		
		Parameters
		----------
		array : NumPy array
			A four-dimensional array containing all the raw data.
			The first dimension is time; the next two dimensions are spacial; the last dimension has length 3 and is colour.
		
		dimensions : pair of integers
			The number of colonies in each direction of the colony array.
	"""
	def __init__(self,array,dimensions=(12,8)):
		self.array = array
		self.format = np.asarray(dimensions,dtype=int)
	
	def segment(self):
		"""
		Tries to automatically segment the images into individual colonies.
		"""
		pass

