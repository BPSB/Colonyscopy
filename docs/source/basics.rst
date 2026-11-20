
Basics
======

Installation
------------

Install Colonyscopy from the project directory:

.. code-block:: bash

   python -m pip install .


Image requirements
------------------

Plates
~~~~~~

Images should show colonies arranged in a regular rectangular grid, with consistent spacing between colony rows and columns.

The entire colony grid should be visible, and colony positions should remain consistent throughout the image time series.

For reliable analysis, images should have consistent dimensions, lighting conditions, and camera positioning.


Image formats
~~~~~~~~~~~~~

Colonyscopy supports the following image formats:

- JPEG (``.jpg``, ``.jpeg``)
- PNG (``.png``)
- TIFF (``.tif``, ``.tiff``)
- Canon RAW (``.cr2``)

All images in a time series must have matching dimensions and data types.

Images should be provided in chronological order to ensure correct analysis of colony growth over time.
