# Great Circle Path

A simple script to find pixels along great circle paths.  Supports [Healpix](https://healpy.readthedocs.io/en/latest/index.html) and [MW](https://arxiv.org/abs/1110.6298#:~:text=The%20fundamental%20property%20of%20any,(L%5E2)%20samples.)

## Installation

All dependencies managed by [poetry](https://python-poetry.org/)

```bash
cd GreatCirclePaths
poetry install
source .venv/bin/activate
```

Note for the last line, the location of the environment (in this case `.venv`) will depend on your poetry configuration.  If you prefer to use `pip` the main dependecies are [healpy](https://pypi.org/project/healpy/) and [pyssht](https://pypi.org/project/pyssht/).

Test the code with

```bash
pytest
```

## Usage

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
