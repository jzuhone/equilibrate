![logo](/docs/source/_images/cluster_generator_logo.png)

# Cluster Generator

[![yt-project](https://img.shields.io/static/v1?label=%22works%20with%22&message=%22yt%22&color=%22blueviolet%22)](https://yt-project.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![ncodes](https://img.shields.io/static/v1?label=%22Implemented%20Sim.%20Codes%22&message=%227%22&color=%22red%22)](https://eliza-diggins.github.io/cluster_generator/build/html/codes.html)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://eliza-diggins.github.io/cluster_generator)

![testing](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg)
![Github Pages](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=master)](https://coveralls.io/github/Eliza-Diggins/cluster_generator)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Cluster Generator](https://jzuhone.github.io/cluster_generator) (CG) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CG provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, CG is intended to interface with
the vast majority of N-body / hydrodynamics codes, reducing the headache of converting initial conditions between formats for different simulation software. GCP's goal is to provide
comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

This repository contains the core package, which is constructed modularly to facilitate easy development by users to meet particular scientific use cases. All of the
necessary tools to get started building initial conditions are provided.

You can access the documentation [here](http:jzuhone.github.io/cluster_generator), or build it from scratch using the `./docs` directory in this source distribution.

Development occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on
the issues page of the repository.

For installation directions, continue reading this README, or visit the [getting started page](https://jzuhone.github.io/cluster_generator/build/html/Getting_Started.html).

## Contents

- [Getting The Package](#getting-the-package)
  - [From PyPI](#from-pypi)
  - [With Conda](#with-conda)
  - [With PIP](#with-pip)
  - [From Source](#from-source)
  - [Dependencies](#dependencies)
- [Contributing Code, Documentation, or Feedback](#contributing-code-documentation-or-feedback)
- [License](#licence)

______________________________________________________________________

## Getting the Package

The `cluster_generator` package can be obtained for python versions 3.8 and up. Installation instructions are provided
below for installation from source code, from `pip` and from `conda`.

### From PyPI

> \[!IMPORTANT\]
> This feature is not yet available.

### With Conda

> \[!IMPORTANT\]
> This feature is not yet available.

### With PIP

> \[!IMPORTANT\]
> This feature is not yet available.

### From Source

To install the library directly from source code, there are two options. If you are using / have installed pip, you can
install directly from the Github URL as follows:

```bash
pip install git+https://www.github.com/eliza-diggins/cluster_generator
```

This will then clone the repository into your path libraries for the python environment you are using and run the setup procedure to install
the software. You can check for a successful installation using `pip show cluster_generator`.

Additionally, if you'd like to install the package for development, this process can be carried out in two steps. First,

```
git clone https://www.github.com/jzuhone/cluster_generator
```

to clone the repository, then

```
pip install . -e
```

to install the package in developer mode.

### Dependencies

`cluster_generator` is compatible with Python 3.9+, and requires the following
Python packages:

- [unyt](http://unyt.readthedocs.org%3E) \[Units and quantity manipulations\]
- [numpy](http://www.numpy.org) \[Numerical operations\]
- [scipy](http://www.scipy.org) \[Interpolation and curve fitting\]
- [h5py](http://www.h5py.org%3E) \[h5 file interaction\]
- [tqdm](https://tqdm.github.io) \[Progress bars\]
- [ruamel.yaml](https://yaml.readthedocs.io) \[yaml support\]
- [dill](https://github.com/uqfoundation/dill) \[Serialization\]
- [halo](https://github.com/manrajgrover/halo) \[Progress Spinners\]
- [pandas](https://github.com/pandas-dev/pandas) \[Dataset Manipulations\]

These will be installed automatically if you use `pip` or `conda` as detailed below.

Though not required, it may be useful to install [yt](https://yt-project.org)
for creation of in-memory datasets from `cluster_generator` and/or analysis of
simulations which are created using initial conditions from
`cluster_generator`.

## Contributing Code Documentation or Feedback

All contributions, bug fixes, documentation improvements, and ideas are welcome. If you're interested in pursuing further development of
Cluster Generator, we suggest you start by browsing the [API Documentation](https://jzuhone.github.io/cluster_generator/build/html/api.html). When you're ready
create a fork of this branch and begin your development. When you finish,
feel free to  add a pull request to this repository and we will review your code contribution.

## Licence
