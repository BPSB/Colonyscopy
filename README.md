# Colonyscopy

A Python package for analysing bacterial colony growth.

Colonyscopy processes images of colonies arranged in a rectangular grid and measures changes in colony intensity over time. It provides tools for automatic image cropping, colony detection and growth analysis using time-series images.

Colonyscopy provides functions for:

- Loading image time series in CR2, JPEG, PNG and TIFF formats.
- Automatically cropping images using Fourier analysis.
- Detecting and excluding unwanted image features using spatial and temporal masks.
- Identifying colony positions.
- Measuring colony intensity over time.

### Documentation

The documentation includes installation instructions, usage examples and an API reference and can be built locally using Sphinx:

```bash
make -C docs html
```

The generated HTML documentation can then be found in docs/build/html/index.html.
To view the generated documentation in a browser, start a local server:

```bash
python -m http.server 8000 --directory docs/build/html
```

### Prerequisites

Colonyscopy requires Python 3.10 or newer and the pip package manager.
The required Python dependencies are installed automatically when installing the package.

### Installing

Clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install Colonyscopy using pip:

```bash
python -m pip install .
```

## Testing

Colonyscopy uses pytest for automated testing.

Install pytest if necessary:

```bash
python -m pip install pytest
```

Run the tests from the project directory:

```bash
python -m pytest
```
