# Cluster Generator

[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)
[![ncodes](https://img.shields.io/static/v1?label="Implemented%20Sim.%20Codes"&message="7"&color="red")](https://eliza-diggins.github.io/cluster_generator/build/html/codes.html)
[![docs]( https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://eliza-diggins.github.io/cluster_generator)
![testing](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg)
![Pylint](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg)
![Github Pages](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=master)](https://coveralls.io/github/Eliza-Diggins/cluster_generator)

The [Cluster Generator Project](https://eliza-diggins.github.io/cluster_generator) (CGP) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CGP provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, the CGP is intented to interface with 
the vast majority of N-body / hydrodynamics codes, reducing the headache of converting initial conditions between formats for different simulation softwares. GCP's goal is to provide
comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

This repository contains the core package, which is constructed modularly to facilitate easy development by users to meet particular scientific use cases. All of the 
necessary tools to get started building initial conditions are provided.

You can access the documentation [here](http:eliza-diggins.github.io/cluster_generator), or build it from scratch using the ``./docs`` directory in this source distribution.

Developement occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on 
the issues page of the repository.

For installation directions, continue reading this README, or visit the [getting started page](https://eliza-diggins.github.io/cluster_generator/build/html/Getting_Started.html).


## Contents

- [Getting The Package](#Getting-the-Package)
  - [From PyPI](#From-PyPI)
  - [With Conda](#With-Conda)
  - [With PIP](#With-PIP)
  - [From Source](#From-Source)
  - [Dependencies](#dependencies)
- [Contributing Code, Documentation, or Feedback](#Contributing-Code-Documentation-or-Feedback)
- [License](#licence)
---

## Getting the Package

The ``cluster_generator`` package can be obtained for python versions 3.8 and up. Installation instructions are provided
below for installation from source code, from ``pip`` and from ``conda``.

### From PyPI

> [!IMPORTANT]  
> This feature is not yet available.


### With Conda

> [!IMPORTANT]  
> This feature is not yet available.

### With PIP

> [!IMPORTANT]  
> This feature is not yet available.

### From Source

To install the library directly from source code, there are two options. If you are using / have installed pip, you can 
install directly from the github URL as follows:

- Using your preferred environment (venv, local python installation, etc), call

  ```bash
  pip install git+https://www.github.com/eliza-diggins/cluster_generator
  ```
  This will install directly from this repository without generating a local clone.
- If you're interested in having a local clone, you can instead do the following
  - First, clone the repository using
    ```bash
    git clone https://www.github.com/eliza-diggins/cluster_generator
    ```
    
    > [!WARNING]  
    > Make sure to navigate to a directory where you want the clone to appear.

    Once the clone has been generated, change your directory so that you are inside the clone and in the same directory as the ``setup.py`` script. Then run the following command:
    
    ```bash
    pip install .
    ```
    This will install the local clone to your python installations ``site-packages`` directory. If you want to install the package in place, you can use
    ```bash
    pip install -e .
    ```
    which will install the package in development mode.

    > [!WARNING]  
    > If the package is installed in development mode, it will not be generically available from any directory.

To test that you've installed the project, simply run
```bash
pip show cluster_generator
```

### Dependencies

``cluster_generator`` is compatible with Python 3.8+, and requires the following
Python packages:

- [unyt](http://unyt.readthedocs.org>) [Units and quantity manipulations]
- [numpy](http://www.numpy.org) [Numerical operations]
- [scipy](http://www.scipy.org) [Interpolation and curve fitting]
- [h5py](http://www.h5py.org>) [h5 file interaction]
- [tqdm](https://tqdm.github.io) [Progress bars]
- [ruamel.yaml](https://yaml.readthedocs.io) [yaml support]
- [dill](https://github.com/uqfoundation/dill) [Serialization]
- [halo](https://github.com/manrajgrover/halo) [Progress Spinners]
- [pandas](https://github.com/pandas-dev/pandas) [Dataset Manipulations]

These will be installed automatically if you use ``pip`` or ``conda`` as detailed below.


Though not required, it may be useful to install [yt](https://yt-project.org)
for creation of in-memory datasets from ``cluster_generator`` and/or analysis of
simulations which are created using initial conditions from
``cluster_generator``.

## Contributing Code Documentation or Feedback

All contributions, bug fixes, documentation improvements, and ideas are welcome. If you're interested in pursuing further development of the
Cluster Generator Project, we suggest you start by browsing the [API Documentation](https://eliza-diggins.github.io/cluster_generator/build/html/api.html). When you're ready
create a fork of this branch and begin your development. When you finish,
feel free to  add a pull request to this repositiory and we will review your code contribution.

## Licence

