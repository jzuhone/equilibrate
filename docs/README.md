# Cluster Generator

---
[![yt-project](https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet")](https://yt-project.org)
[![docs]( https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://eliza-diggins.github.io/cluster_generator)
![testing](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg)
![Pylint](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg)
![Github Pages](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=master)](https://coveralls.io/github/Eliza-Diggins/cluster_generator?branch=MOND)

The [Cluster Generator Project](https:eliza-diggins.github.io/cluster_generator) (CGP) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CGP provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, the CGP is intented to interface with 
the vast majority of N-body / hydrodynamics codes, reducing the headache of converting initial conditions between formats for different simulation softwares. GCP's goal is to provide
comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

This repository contains the core package, which is constructed modularly to facilitate easy development by users to meet particular scientific use cases. All of the 
necessary tools to get started building initial conditions are provided.

You can access the documentation [here](http:eliza-diggins.github.io/cluster_generator), or build it from scratch using the ``./docs`` directory in this source distribution.

Developement occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on 
the issues page of the repository.

For installation directions, continue reading this README, or visit the [getting started page](http:eliza-diggins.github.io/cluster_generator/Getting_Started).


## Contents

---

- [Getting The Package](#Getting-the-Package)
  - [From PyPI](#From-PyPI)
  - [With Conda](#With-Conda)
  - [With PIP](#With-PIP)
  - [Dependencies](#dependencies)
- [Contributing Code, Documentation, or Feedback](#Contributing-Code-Documentation-or-Feedback)
- [License](#licence)
---

## Getting the Package

The ``cluster_generator`` package can be obtained for python versions 3.6 and up. Installation instructions are provided
below for installation from source code, from ``pip`` and from ``conda``.

### From PyPI

> [!IMPORTANT]  
> This feature is not yet available.

### From Source

To gather the necessary code from source, simple navigate to a directory in which you'd like to store the local copy
of the package and execute

```bash
    git clone https://github.com/jzuhone/cluster_generator
```

### With Conda

### With PIP


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
