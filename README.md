[![PyPI version](https://badge.fury.io/py/greatcirclepaths.svg)](https://badge.fury.io/py/greatcirclepaths)

# Great Circle Path

A simple package and script to find pixels along great circle paths.  Supports [Healpix](https://healpy.readthedocs.io/en/latest/index.html) and [MW](https://arxiv.org/abs/1110.6298#:~:text=The%20fundamental%20property%20of%20any,(L%5E2)%20samples.)

## Installation

Package can be found on [pypi](https://pypi.org/project/greatcirclepaths/).

```bash
pip install greatcirclepaths
```

If installing from source, all dependencies can be managed by [poetry](https://python-poetry.org/)

```bash
cd GreatCirclePaths
poetry install
source .venv/bin/activate
```

Note for the last line, the location of the environment (in this case `.venv`) will depend on your poetry configuration.

Test the code with

```bash
pytest
```

## Usage

Main script can be used as

```bash
cd greatcirclepaths
python main.py <infile> <outfile> <sampling> <options>
```

`<infile>` is a file path containing the colatitude and longitude in radians of the start and end points of the great circle paths, with columns

> start_colatitude,  start_longitude, end_colatitude, end_longitude

If the data is i lat/lon (degrees) use the `--latlon` flag.

`<outfile>` is a file path where output will be saved as a `.npz` file.  Output is a `scipy.sparse` matrix, where each row is a HealPix map with 1s along the great circle path.

`<sampling>` must be either `MW` or `hpx`.

Run `python main.py --help` for further details.

This can be quite slow when picking many paths so we recommend using as many processes as possible.  Default is 4.

The package also gives access to the `GreatCirclePath` class for things like finding the lenght of paths and coordinates along the path.

## Visulaisation

Manipulate the sparse matrix as you want and convert to a regular `numpy` array.  Then just use the [healpy visulaisation functions](https://healpy.readthedocs.io/en/latest/healpy_visu.html). For example

```python
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import healpy as hp
from scipy import sparse

path_matrix = sparse.load_npz(<outfile>)
path_map = np.array(path_matrix.sum(axis=0))[0]
hp.mollview(path_map, flip="geo", cmap=cm.jet, title="Ray Density", unit="Number of Rays")
```

![Alt text](example.png)
