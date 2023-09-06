# Cluster Generator

---
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)
[![docs]( https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://eliza-diggins.github.io/cluster_generator)
![testing](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg)
![Pylint](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg)
![Github Pages](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=master)](https://coveralls.io/github/Eliza-Diggins/cluster_generator?branch=MOND)

``cluster_generator`` is a cross-platform galaxy cluster initializer for N-body / hydrodynamics codes. ``cluster_generator`` supports
a variety of different possible configurations for the initialized galaxy clusters, including a variety of profiles, different construction
assumptions, and non-Newtonian gravity options.

## Getting the Package

### From PyPI

[Not yet implemented]

### From Source

To gather the necessary code from source, simple navigate to a directory in which you'd like to store the local copy
of the package and execute

```bash
    git clone https://github.com/jzuhone/cluster_generator
```


If you want a specific branch of the project, use the ``-b`` flag in the command and provide the name of the branch.

Once the git clone has finished, there should be a directory ``./cluster_generator`` in your current working directory.

### Dependencies

``cluster_generator`` is compatible with Python 3.8+, and requires the following
Python packages:

- [unyt](http://unyt.readthedocs.org>) [Units and quantity manipulations]
- [numpy](http://www.numpy.org) [Numerical operations]
- [scipy](http://www.scipy.org) [Interpolation and curve fitting]
- [h5py](http://www.h5py.org>) [h5 file interaction]
- [tqdm](https://tqdm.github.io) [Progress bars]
- [ruamel.yaml](https://yaml.readthedocs.io) [yaml support]

These will be installed automatically if you use ``pip`` or ``conda`` as detailed below.


Though not required, it may be useful to install [yt](https://yt-project.org)
for creation of in-memory datasets from ``cluster_generator`` and/or analysis of
simulations which are created using initial conditions from
``cluster_generator``.
