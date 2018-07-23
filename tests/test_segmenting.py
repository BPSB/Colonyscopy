from pytest import raises
import numpy as np
from colonyscopy import ColonyArray
from generate_data import generate_data,colony_centres

times = 10
dimensions = (12,8)
colony_sizes = np.random.uniform(10,100,(times,*dimensions))
resolution = (900,600)
bg = np.random.randint(0,256,3)
fg = np.random.randint(0,256,3)

some_array = ColonyArray(
		array = generate_data(colony_sizes,resolution,bg,fg),
		dimensions = dimensions
	)

centres = colony_centres(dimensions,resolution)

for i in range(dimensions[0]):
	for j in range(dimensions[1]):
		assert centres[i,j] == some_array[i,j].centre

for i,j in [ (dimensions[0],0), (0,dimensions[1]) ]:
	with raises(IndexError):
		centres[i,j]

