import numpy as np
from colonyscopy._colonyscopy import Colony, MaskComputationFailed
import pytest

size = (41, 41)
times = 10
ncolours = 3
centre = (20, 20)

@pytest.fixture
def dummy_colony():
    return Colony(
        images=np.zeros((times, *size, ncolours)),
        background=np.zeros((*size, ncolours)),
        speckle_mask=np.ones(size, dtype=bool),
        centre=centre
    )


class TestTempMean(object):
    def test_calculates_mean_over_time(self):
        images = np.array([np.full((*size, ncolours), 2),np.full((*size, ncolours), 4),np.full((*size, ncolours), 6),])

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre,
        )

        expected = np.full((*size, ncolours), 4)
        np.testing.assert_array_equal(dummy_colony.temp_mean, expected)

    def test_returns_cached_mean(self, dummy_colony):
        first_mean = dummy_colony.temp_mean
        second_mean = dummy_colony.temp_mean

        assert first_mean is second_mean


class TestColonyIntensity(object):
    def test_length(self, dummy_colony):
        a = dummy_colony.colony_intensity

        assert a.shape == (times,)

    def test_intensity_increase(self):
        intensity_of_point = 10000
        timepoint = 7
        images = np.zeros((times, *size, ncolours), dtype=np.float64)
        images[timepoint:, centre[0], centre[1], 0] = intensity_of_point

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre
        )

        a = dummy_colony.colony_intensity

        np.testing.assert_allclose(a[:timepoint], 0, atol=1e-8)
        assert np.all(a[timepoint:] > a[:timepoint].max())

    def test_colony_background_difference(self):
        images = np.full((times, *size, ncolours),10.0,dtype=np.float64)

        y, x = np.indices(size)
        colony_region = (y-centre[0])**2 +(x-centre[1])**2 <= 10**2
        images[:, colony_region, :] = 20.0

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre
        )

        result = dummy_colony.colony_intensity
        expected = np.full(times, 30.0)

        np.testing.assert_allclose(result, expected)

    def test_recursion_error(self, monkeypatch):
        images = np.full((times, *size, ncolours), 100.0)
        images[:, 10:31, 10:31, :] = 0

        point = (20, 39)
        images[:, point[0], point[1], :] = 10000

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre
        )

        original_method = Colony.create_colony_intensity
        calls = 0

        def fake_method(self):
            nonlocal calls
            calls += 1

            if calls == 2:
                raise RecursionError

            return original_method(self)

        monkeypatch.setattr(Colony,"create_colony_intensity",fake_method)
        dummy_colony.create_colony_intensity()

        assert calls == 2
        assert dummy_colony._colony_intensity.shape == (times,)
        assert np.all(np.isfinite(dummy_colony._colony_intensity))


    def test_empty_colony_mask(self, dummy_colony):
        dummy_colony.speckle_mask[:] = False
        intensity = dummy_colony.colony_intensity

        np.testing.assert_array_equal(intensity, np.zeros(times))
        assert np.all(np.isfinite(intensity))

    def test_empty_background_mask(self, dummy_colony):
        dummy_colony.background_mask = np.zeros(size, dtype=bool)
        intensity = dummy_colony.colony_intensity

        np.testing.assert_array_equal(intensity, np.zeros(times))
        assert np.all(np.isfinite(intensity))


class TestColonyAreaMask(object):
    def test_format(self, dummy_colony):

        dummy_mask = dummy_colony.mask
        assert dummy_mask.shape == size
        assert dummy_mask.dtype == bool

    def test_contains_centre(self, dummy_colony):
        assert dummy_colony.mask[centre]

    def test_mask_excludes_speckles(self):
        speckle_mask = np.ones(size, dtype=bool)
        speckle_mask[centre] = False

        dummy_colony = Colony(
        images = np.zeros([times,*size,ncolours]),
        background = np.zeros((*size,ncolours)),
        speckle_mask = speckle_mask,
        centre=centre)

        assert not dummy_colony.mask[centre]

    def test_mask_respects_colony_radius(self, dummy_colony):
        dummy_colony.create_mask(colony_mask_width=3)
        dummy_mask = dummy_colony.mask

        assert dummy_mask[20, 20]
        assert dummy_mask[20, 23]
        assert not dummy_mask[20, 24]
        assert not dummy_mask[23, 23]


class TestSegmentIntensity(object):
    def test_one_intensity_per_timepoint(self, dummy_colony):
        a = dummy_colony.segment_intensity()
        assert a.size == times
        assert a.shape == (times,)

    def test_point_constant_intensity(self):
        intensity_of_point = 10000
        point = centre
        colour = 0
        timepoint = 7

        images = np.zeros((times,*size,ncolours), dtype=np.uint16)
        images[timepoint:, point[0], point[1], colour] = intensity_of_point

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre
        )

        a = dummy_colony.segment_intensity()
        raw_intensity = intensity_of_point / np.prod(size)
        expected = np.zeros(times)
        expected[timepoint:] = raw_intensity

        np.testing.assert_allclose(a, expected)

    def test_baseline_correction(self):
        baseline = 100
        intensity_increase = 10000
        timepoint = 7

        images = np.full((times, *size, ncolours), baseline, dtype=np.float64)
        images[timepoint:, centre[0], centre[1], 0] += intensity_increase

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre
        )
        a = dummy_colony.segment_intensity()
        expected = np.zeros(times)
        expected[timepoint:] = intensity_increase / np.prod(size)

        np.testing.assert_allclose(a, expected)

    def test_empty_speckle_mask(self, dummy_colony):
        dummy_colony.speckle_mask[:] = False

        with pytest.raises(MaskComputationFailed, match="No valid pixels"):
            dummy_colony.segment_intensity()



class TestBackgroundMask(object):
    def test_format(self, dummy_colony):
        dummy_mask = dummy_colony.background_mask

        assert dummy_mask.shape == size
        assert dummy_mask.dtype == bool

    def test_excludes_colony_region(self, dummy_colony):
        dummy_colony.create_background_mask(bg_mask_width=18)
        dummy_mask = dummy_colony.background_mask

        assert not dummy_mask[centre]
        assert not dummy_mask[20, 38]
        assert dummy_mask[20, 39]

    def test_excludes_bright_background_pixel(self):
        images = np.full((times, *size, ncolours), 100, dtype=np.float32)
        images[:, 10:31, 10:31, :] = 0
        point = (20, 39)
        images[:, point[0], point[1], :] = 10000

        dummy_colony = Colony(
            images=images,
            background=np.zeros((*size, ncolours)),
            speckle_mask=np.ones(size, dtype=bool),
            centre=centre,
        )

        _ = dummy_colony.colony_intensity

        assert not dummy_colony.background_mask[point]

    def test_set_valid_mask(self, dummy_colony):
        new_mask = np.ones(size, dtype=bool)
        new_mask[centre] = False
        dummy_colony.background_mask = new_mask

        np.testing.assert_array_equal(dummy_colony.background_mask, new_mask)

    def test_reject_wrong_shape(self, dummy_colony):
        original_mask = dummy_colony.background_mask.copy()
        wrong_mask = np.ones((20, 20), dtype=bool)
        dummy_colony.background_mask = wrong_mask

        np.testing.assert_array_equal(dummy_colony.background_mask, original_mask)

    def test_reject_wrong_dtype(self, dummy_colony):
        original_mask = dummy_colony.background_mask.copy()
        wrong_mask = np.ones(size, dtype=float)
        dummy_colony.background_mask = wrong_mask

        np.testing.assert_array_equal(dummy_colony.background_mask, original_mask)


class TestColonyRadialProfile(object):
    def test_output_shape(self, dummy_colony):
        bins, values = dummy_colony.colony_radial_profile

        assert bins.shape == (60,)
        assert values.shape == (60,)

    def test_constant_image(self, dummy_colony):
        dummy_colony.images[-1] = 10
        bins, values = dummy_colony.colony_radial_profile

        np.testing.assert_allclose(values, 30)

    def test_returns_cached_profile(self, dummy_colony):
        first_profile = dummy_colony.colony_radial_profile
        second_profile = dummy_colony.colony_radial_profile

        assert first_profile is second_profile

    def test_custom_number_of_points(self, dummy_colony):
        dummy_colony.create_radial_profile(npoints=25)
        bins, values = dummy_colony.colony_radial_profile

        assert bins.shape == (25,)
        assert values.shape == (25,)