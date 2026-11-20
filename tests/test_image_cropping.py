import numpy as np
import pytest
from colonyscopy._colonyscopy import Image_cropping, Plate, ColonyscopyFailedHeuristic
from tests.generate_data import generate_data, colony_centres

@pytest.fixture
def images():
    return np.zeros((5, 100, 120, 3))

@pytest.fixture
def cropper(images):
    plate = Plate(images=images, layout=(3, 4), bg=np.zeros((100, 120, 3)))
    return Image_cropping(plate=plate)

@pytest.fixture
def detected_cropper(cropper):
    cropper.fourier_results = {"Rows": {"positions": np.array([20., 40., 60.]), "spacing": 20., "matched_count": 3},
	"Columns": {"positions": np.array([30., 50., 70., 90.]), "spacing": 20., "matched_count": 4},}
    return cropper

@pytest.fixture
def realistic_cropper():
	layout = (12, 8)
	resolution = (900, 600)
	colony_sizes = np.full((3, *layout), 10.0)

	images = generate_data(
		colony_sizes=colony_sizes,
		resolution=resolution,
		bg=(0, 0, 0),
		fg=(100, 100, 100),
		noise_amplitude=0,
		rim_width=0,
	)

	plate = Plate(images=images, layout=layout, bg=np.zeros((*resolution, 3)))
	return Image_cropping(plate=plate), colony_centres(layout, resolution)

class TestInitialization:
    def test_initial_state(self, cropper):
        np.testing.assert_array_equal(cropper.layout, (3, 4))
        assert cropper.fourier_results is None
        assert cropper.fourier_quality is None
        assert cropper.cropped_image is None
        assert cropper.crop_bounds is None
        assert cropper.crop_approved is False


class TestDetectGridFourier:
    def test_result_format(self, realistic_cropper):
        cropper, _ = realistic_cropper
        results = cropper.detect_grid_fourier(plot=False)
        assert cropper.fourier_results is results
        for axis, expected in (("Rows", 12), ("Columns", 8)):
            assert set(results[axis]) == {"spacing", "positions", "matched_count"}
            assert len(results[axis]["positions"]) == expected
            assert isinstance(results[axis]["matched_count"], int)
            assert 0 <= results[axis]["matched_count"] <= expected
            assert results[axis]["spacing"] > 0

    def test_realistic_colony_positions(self, realistic_cropper):
        cropper, centres = realistic_cropper
        results = cropper.detect_grid_fourier(plot=False)
        np.testing.assert_allclose(results["Rows"]["positions"], centres[:, 0, 0], atol=2)
        np.testing.assert_allclose(results["Columns"]["positions"], centres[0, :, 1], atol=2)

    def test_no_fourier_peaks(self, cropper):
        with pytest.raises(ColonyscopyFailedHeuristic, match="no Fourier peaks detected"):
            cropper.detect_grid_fourier(plot=False)


class TestFourierQuality:
    @pytest.mark.parametrize("row_matches,column_matches,passed", [(3, 4, True), (2, 4, False), (3, 2, False), (2, 2, False),])

    def test_grid_positions_matched(self, detected_cropper, row_matches, column_matches, passed):
        detected_cropper.fourier_results["Rows"]["matched_count"] = row_matches
        detected_cropper.fourier_results["Columns"]["matched_count"] = column_matches
        quality = detected_cropper._evaluate_fourier_quality()
        assert detected_cropper.fourier_quality is quality
        assert quality["Rows"] == {"matched": row_matches, "expected": 3, "passed": row_matches == 3}
        assert quality["Columns"] == {"matched": column_matches, "expected": 4, "passed": column_matches == 4}
        assert bool(quality["all_passed"]) is passed


class TestFourierBounds:
    def test_half_spacing_margin(self, cropper):
        assert cropper._get_fourier_bounds(np.array([20, 40, 60]), 20, 100) == (10, 70)

    def test_rounds_crop_boundaries(self, cropper):
        assert cropper._get_fourier_bounds(np.array([20.2, 60.2]), 19, 100) == (10, 70)

    def test_clipped_to_image(self, cropper):
        assert cropper._get_fourier_bounds(np.array([5, 95]), 20, 100) == (0, 100)

    def test_invalid_bounds(self, cropper):
        with pytest.raises(ValueError, match="Invalid crop boundaries"):
            cropper._get_fourier_bounds(np.array([120, 130]), 10, 100)


class TestCropFourier:
    def test_requires_detection(self, cropper):
        with pytest.raises(ValueError, match="detect_grid_fourier"):
            cropper.crop_fourier()

    def test_bounds_and_shape(self, detected_cropper):
        cropped = detected_cropper.crop_fourier()
        assert detected_cropper.crop_bounds == (10, 70, 20, 100)
        assert cropped.shape == (5, 60, 80, 3)
        assert cropped is detected_cropper.cropped_image

    def test_same_crop_for_every_frame(self, detected_cropper):
        for t in range(5):
            detected_cropper.plate.images[t] = t + 1
        cropped = detected_cropper.crop_fourier()
        for t in range(5):
            assert np.all(cropped[t] == t + 1)

    def test_returns_copy(self, detected_cropper):
        cropped = detected_cropper.crop_fourier()
        cropped[:] = 255
        assert np.all(detected_cropper.plate.images == 0)


class TestAutoCropFourier:
    def test_automatic_acceptance(self, detected_cropper, monkeypatch):
        monkeypatch.setattr(detected_cropper, "detect_grid_fourier", lambda plot: detected_cropper.fourier_results)
        cropped = detected_cropper.auto_crop_fourier(plot=False)
        assert cropped is detected_cropper.cropped_image
        assert cropped.shape == (5, 60, 80, 3)
        assert detected_cropper.crop_approved is True
        assert bool(detected_cropper.fourier_quality["all_passed"]) is True

    def test_failed_quality_requests_manual_review(self, detected_cropper, monkeypatch):
        detected_cropper.fourier_results["Rows"]["matched_count"] = 2
        monkeypatch.setattr(detected_cropper, "detect_grid_fourier", lambda plot: detected_cropper.fourier_results)
        called = []
        monkeypatch.setattr(detected_cropper, "manual_fourier_quality", lambda: called.append(True))
        result = detected_cropper.auto_crop_fourier(plot=False)
        assert result is None
        assert called == [True]
        assert detected_cropper.crop_approved is False
        assert bool(detected_cropper.fourier_quality["all_passed"]) is False

    def test_resets_previous_crop(self, detected_cropper, monkeypatch):
        detected_cropper.crop_fourier()
        detected_cropper.crop_approved = True
        detected_cropper.fourier_results["Rows"]["matched_count"] = 2
        monkeypatch.setattr(detected_cropper, "detect_grid_fourier", lambda plot: None)
        monkeypatch.setattr(detected_cropper, "manual_fourier_quality", lambda: None)
        detected_cropper.auto_crop_fourier(plot=False)
        assert detected_cropper.crop_bounds is None
        assert detected_cropper.cropped_image is None
        assert detected_cropper.crop_approved is False


class TestManualFourierQuality:
    @pytest.mark.parametrize("answer,accepted", [("y", True), ("yes", True), ("n", False)])
    def test_manual_acceptance_and_rejection(self, detected_cropper, monkeypatch, answer, accepted):
        import matplotlib.pyplot as plt
        monkeypatch.setattr("builtins.input", lambda _: answer)
        monkeypatch.setattr(plt, "show", lambda: None)
        result = detected_cropper.manual_fourier_quality()
        assert detected_cropper.crop_approved is accepted
        if accepted:
            assert result is detected_cropper.cropped_image
            assert detected_cropper.crop_bounds == (10, 70, 20, 100)
        else:
            assert result is None
            assert detected_cropper.cropped_image is None
            assert detected_cropper.crop_bounds is None
        plt.close("all")

    def test_adjustment_is_requested(self, detected_cropper, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr("builtins.input", lambda _: "a")
        monkeypatch.setattr(plt, "show", lambda: None)
        calls = []
        monkeypatch.setattr(detected_cropper, "adjust_fourier_crop", lambda: calls.append(True))
        assert detected_cropper.manual_fourier_quality() is None
        assert calls == [True]
        plt.close("all")


class TestAdjustFourierCrop:
    @pytest.fixture
    def controls(self, detected_cropper, monkeypatch):
        import colonyscopy._colonyscopy as module
        detected_cropper.crop_fourier()
        monkeypatch.setattr(module, "display", lambda *args, **kwargs: None)
        detected_cropper.adjust_fourier_crop()
        yield detected_cropper._crop_widgets
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_initial_state(self, detected_cropper, controls):
        top, bottom, left, right, *_ = controls
        assert (top.value, bottom.value, left.value, right.value) == (10, 70, 20, 100)
        assert detected_cropper.cropped_image is None
        assert detected_cropper.crop_approved is False

    def test_accept_adjusted_crop(self, detected_cropper, controls):
        top, bottom, left, right, accept, *_ = controls
        top.value, bottom.value, left.value, right.value = 15, 65, 25, 95
        accept.click()
        assert detected_cropper.crop_bounds == (15, 65, 25, 95)
        assert detected_cropper.cropped_image.shape == (5, 50, 70, 3)
        assert detected_cropper.crop_approved is True

    def test_invalid_boundaries_not_accepted(self, detected_cropper, controls):
        top, bottom, left, right, accept, *_ = controls
        top.value = bottom.value
        accept.click()
        assert detected_cropper.crop_approved is False
        assert detected_cropper.cropped_image is None

    def test_reset_fields(self, controls):
        top, bottom, left, right, _, reset, *_ = controls
        top.value, bottom.value, left.value, right.value = 15, 65, 25, 95
        reset.click()
        assert (top.value, bottom.value, left.value, right.value) == (10, 70, 20, 100)
