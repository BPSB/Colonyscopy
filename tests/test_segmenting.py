from pytest import mark
import numpy as np
from numpy.testing import assert_allclose
from colonyscopy._colonyscopy import Plate, Colony
from tests.generate_data import generate_data,colony_centres
from colonyscopy.tools import smoothen_image

times = 10
layout = (12,8)
resolution = (900,600)

rng = np.random.default_rng(42) #reproducible random generator -> colonies with increasing size
colony_sizes = np.dstack([rng.uniform(i+1,4*(i+1),layout) for i in range(times) ]).transpose(2,0,1)

bg = np.array([30, 40, 50])
fg = np.array([180, 190, 200])
rim_color = np.array([220, 220, 220])
rim_width = 5
noise_width = 20

bg_img = np.empty((*resolution,3),dtype=np.uint8)

for c in range(3):
	middle = bg[c]//2+fg[c]//2
	low, high = sorted((int(bg[c]), int(middle)))

	if high == low:
		high += 1

	bg_img[:,:,c] = np.random.randint(low,high,size=resolution)

bg_img = smoothen_image(bg_img,20)
@mark.parametrize("noise_amp, tol",[(0,3),(30,3),])#with or without noise amplitude but the same tolerance
@mark.parametrize("background",[bg,bg_img]) #uniform and varying background

def test_segmenting(noise_amp,tol,background):
	some_array = Plate(
			images = generate_data(
					colony_sizes,
					resolution,
					background,
					fg,
					noise_amp,
					noise_width,
					rim_width,
					rim_color),
			layout = layout,
			bg=background,
		)

	centres_control = colony_centres(layout,resolution)
	assert_allclose(some_array.centres,centres_control,atol=3)

	for i in (0,1):
		borders_control = np.linspace(0,resolution[i],layout[i]+1)
		assert_allclose(some_array.borders[i],borders_control,atol=tol)

def test_coordinates_match_grid_positions():
    images = generate_data(
        colony_sizes,
        resolution,
        bg,
        fg,
        noise_amplitude=0,
        rim_width=rim_width,
        rim_color=rim_color
    )

    dummy_plate = Plate(images=images,layout=layout,bg=bg)

    rows, columns = dummy_plate.coordinates
    expected_rows = np.linspace(resolution[0]/(2*layout[0]),resolution[0] - resolution[0]/(2*layout[0]),layout[0])
    expected_columns = np.linspace(resolution[1]/(2*layout[1]),resolution[1] - resolution[1]/(2*layout[1]),layout[1])

    np.testing.assert_allclose(rows, expected_rows, atol=3)
    np.testing.assert_allclose(columns, expected_columns, atol=3)

def test_colony_centres():
	images = generate_data(
		colony_sizes,
		resolution,
		bg,
		fg,
		noise_amplitude=0,
		rim_width=rim_width,
		rim_color=rim_color
	)
	dummy_plate = Plate(images=images,layout=layout,bg=bg)
	local_centres = dummy_plate.colony_centres

	assert local_centres.shape == (*layout,2)

	for i in range(layout[0]):
		for j in range(layout[1]):
			segment_height = dummy_plate.borders[0][i+1]-dummy_plate.borders[0][i]
			segment_width = dummy_plate.borders[1][j+1]-dummy_plate.borders[1][j]

			assert 0 <= local_centres[i,j,0]<segment_height
			assert 0 <= local_centres[i,j,1]<segment_width

def test_colony_centres_large_coordinates():
        plate = object.__new__(Plate)
        plate.layout = (1,1)
        plate.resolution = (500,500)
        plate._coordinates = (np.array([350]),np.array([400]))
        plate._borders = (np.array([0,500]),np.array([0,500]))

        assert np.array_equal(plate.colony_centres,np.array([[[350,400]]]))
        assert np.issubdtype(plate.colony_centres.dtype,np.integer)


def test_create_colonies_shape_and_type():
    images = generate_data(
        colony_sizes,
        resolution,
        bg,
        fg,
        noise_amplitude=0,
        rim_width=rim_width,
        rim_color=rim_color
    )

    background = np.broadcast_to(bg,(*resolution,3))
    dummy_plate = Plate(images=images,layout=layout,bg=background)
    dummy_plate.create_speckle_mask()
    colonies = dummy_plate.colonies

    assert colonies.shape == layout
    assert all(isinstance(colony, Colony) for colony in colonies.flat) #everthing is a colony object

    for i in range(layout[0]):
        for j in range(layout[1]):
            colony = colonies[i,j]

            expected_height = dummy_plate.borders[0][i+1]-dummy_plate.borders[0][i]
            expected_width = dummy_plate.borders[1][j+1]-dummy_plate.borders[1][j]

            assert colony.images.shape == (times,expected_height,expected_width,images.shape[-1])

def test_all_colonies():
    images = generate_data(
        colony_sizes,
        resolution,
        bg,
        fg,
        noise_amplitude=0,
        rim_width=rim_width,
        rim_color=rim_color
    )
    background = np.broadcast_to(bg,(*resolution, 3))
    dummy_plate = Plate(images=images,layout=layout,bg=background)
    dummy_plate.create_speckle_mask()
    generator = dummy_plate.all_colonies()

    assert hasattr(generator, "__next__")

    colonies = list(generator)

    assert len(colonies) == layout[0] * layout[1]

    for i in range(layout[0]):
        for j in range(layout[1]):
            index = i * layout[1]+j

            assert colonies[index] is dummy_plate.colonies[i,j]

class TestPlateInitialization:

    def test_no_background(self):
        images = np.zeros((3, 20, 30, 3))
        images[0, :, :, :] = [30, 60, 90]
        images[1, :, :, :] = [100, 120, 140]

        dummy_plate = Plate(images=images, bg=None)

        assert dummy_plate.background.shape == (20, 30, 3)

        expected_background = np.broadcast_to([30, 60, 90],(20, 30, 3))

        np.testing.assert_array_equal(dummy_plate.background,expected_background)