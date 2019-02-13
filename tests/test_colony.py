from pytest import mark
import numpy as np
from colonyscopy import Colony
import colonyscopy.tools

size = (5,5)
times = 5
ncolours = 4

class TestColonyAreaMask(object):
    def test_format(self):
        dummy_colony = Colony(
        images = np.zeros([times,*size,ncolours]),
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones((size), dtype=bool))
        dummy_mask = dummy_colony.mask
        assert dummy_mask.shape == size
        assert dummy_mask.dtype == bool

    def test_single_point(self):
        seg_intensity_threshold = 100
        point = tuple([np.random.randint(x) for x in size])
        colour = np.random.randint(ncolours)
        images = np.zeros((times,*size,ncolours),dtype=np.uint16)
        for t in range(times):
            images[(t,*point,colour)] = 2*seg_intensity_threshold*np.multiply(*size)
        dummy_colony = Colony(
        images = images,
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones(size, dtype=bool))
        dummy_colony.create_threshold_timepoint(seg_intensity_threshold = seg_intensity_threshold)
        dummy_mask = dummy_colony.mask
        print(dummy_mask)
        assert dummy_mask[point]
        assert np.sum(dummy_mask) == 1

class TestSegmentIntensity(object):
    def test_length(self):
        dummy_colony = Colony(
        images = np.zeros([times,*size,ncolours]),
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones((size), dtype=bool))
        a = dummy_colony.segment_intensity()
        assert a.size == times

    def test_point_constant_intensity(object):
        intensity_of_point = 10000
        point = tuple([np.random.randint(x) for x in size])
        colour = np.random.randint(ncolours)
        images = np.zeros((times,*size,ncolours),dtype=np.uint16)
        timepoint = np.random.randint(times-1)
        for t in range(times-timepoint):
            images[(t+timepoint,*point,colour)] = intensity_of_point
        dummy_colony = Colony(
        images = images,
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones(size, dtype=bool))
        a = dummy_colony.segment_intensity()
        print(a)
        assert np.sum(a > 0) == times-timepoint
        assert a[timepoint+1] == intensity_of_point/np.multiply(*size)

class TestColonyIntensity(object):
    def test_length(self):
        intensity_of_point = 100000
        point = tuple([np.random.randint(x) for x in size])
        colour = np.random.randint(ncolours)
        images = np.zeros((times,*size,ncolours),dtype=np.uint16)
        timepoint = np.random.randint(times-1)
        for t in range(times-timepoint):
            images[(t+timepoint,*point,colour)] = intensity_of_point
        dummy_colony = Colony(
        images = images,
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones((size), dtype=bool))
        print(dummy_colony.n_colours)
        a = dummy_colony.colony_intensity()
        assert a.size == times

class TestBackgroundMask(object):
    def test_format(self):
        dummy_colony = Colony(
        images = np.zeros([times,*size,ncolours]),
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones((size), dtype=bool))
        dummy_mask = dummy_colony.background_mask
        assert dummy_mask.shape == size
        assert dummy_mask.dtype == bool

    def test_single_point(self):
        seg_intensity_threshold = 100
        point = tuple([np.random.randint(x) for x in size])
        colour = np.random.randint(ncolours)
        images = np.zeros((times,*size,ncolours),dtype=np.uint16)
        for t in range(times):
            images[(t,*point,colour)] = 2*seg_intensity_threshold*np.multiply(*size)
        dummy_colony = Colony(
        images = images,
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones(size, dtype=bool))
        dummy_colony.create_threshold_timepoint(seg_intensity_threshold = seg_intensity_threshold)
        dummy_mask = dummy_colony.background_mask
        assert not dummy_mask[point]

class TestThresholdTimepoint(object):
    def test_single_point_appearing(self):
        seg_intensity_threshold = 100
        point = tuple([np.random.randint(x) for x in size])
        colour = np.random.randint(ncolours)
        images = np.zeros((times,*size,ncolours),dtype=np.uint16)
        timepoint = np.random.randint(times-1)
        for t in range(times-timepoint):
            images[(t+timepoint,*point,colour)] = 2*seg_intensity_threshold*np.multiply(*size)
        dummy_colony = Colony(
        images = images,
        background = np.zeros((*size,ncolours)),
        speckle_mask = np.ones((size), dtype=bool))
        dummy_colony.create_threshold_timepoint(seg_intensity_threshold = seg_intensity_threshold)
        assert dummy_colony.threshold_timepoint == timepoint
