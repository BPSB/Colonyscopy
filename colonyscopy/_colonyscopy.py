

class ColonyArray(object):
	"""
		The basic class for handling colony arrays.
		
		Parameters
		----------
		array : NumPy array
			An three-dimensional array containing all the raw data.
			The first two dimensions are spacial, the third dimension is time.
		
		dimensions : pair of integers
			The number of colonies in each direction of the colony array.
	"""
	def __init__(array,dimensions=(12,8)):
		self.array = array
		self.format = np.asarray(dimensions,dtype=int)
	
	def segment(self):
		"""
		Tries to automatically segment the images into individual colonies.
		"""
		pass

