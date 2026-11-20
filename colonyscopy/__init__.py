from ._colonyscopy import Plate,Colony,Image_cropping,ColonyscopyFailedHeuristic,MaskComputationFailed
from .tools import load_images,smoothen,smoothen_image,expand_mask,color_sum,color_distance

try:
    from .version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("colonyscopy")
    except PackageNotFoundError:
        __version__ = "unknown"