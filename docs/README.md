![logo](/source/_images/cluster_generator_logo.png)

# The Cluster Generator Project

[![yt-project](https://img.shields.io/static/v1?label=%22works%20with%22&message=%22yt%22&color=%22blueviolet%22)](https://yt-project.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![ncodes](https://img.shields.io/static/v1?label=%22Implemented%20Sim.%20Codes%22&message=%227%22&color=%22red%22)](https://eliza-diggins.github.io/cluster_generator/build/html/codes.html)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://eliza-diggins.github.io/cluster_generator)
![testing](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg)
![Github Pages](https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg)
[![Coverage](https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=master)](https://coveralls.io/github/Eliza-Diggins/cluster_generator)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

The [Cluster Generator Project](https://eliza-diggins.github.io/cluster_generator) (CGP) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CGP provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, the CGP is intended to interface with
the vast majority of N-body / hydrodynamics codes, reducing the headache of converting initial conditions between formats for different simulation software. GCP's goal is to provide
comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

You can access the documentation [here](http:eliza-diggins.github.io/cluster_generator), or build it from scratch using the `./docs` directory in this source distribution.

Development occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on
the issues page of the repository.

For installation directions, continue reading this README, or visit the [getting started page](https://eliza-diggins.github.io/cluster_generator/build/html/Getting_Started.html).

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

<style>

.alert {
  position: relative;
  top: 10;
  left: 0;
  width: auto;
  height: auto;
  padding: 10px;
  margin: 10px;
  line-height: 1.8;
  border-radius: 5px;
  cursor: hand;
  cursor: pointer;
  font-family: sans-serif;
  font-weight: 400;
}

.alertCheckbox {
  display: none;
}

:checked + .alert {
  display: none;
}

.alertText {
  display: table;
  margin: 0 auto;
  text-align: center;
  font-size: 16px;
}

.alertClose {
  float: right;
  padding-top: 5px;
  font-size: 10px;
}

.clear {
  clear: both;
}

.info {
  background-color: #EEE;
  border: 1px solid #DDD;
  color: #999;
}

.success {
  background-color: #EFE;
  border: 1px solid #DED;
  color: #9A9;
}

.notice {
  background-color: #EFF;
  border: 1px solid #DEE;
  color: #9AA;
}

.warning {
  background-color: #FDF7DF;
  border: 1px solid #FEEC6F;
  color: #C9971C;
}

.error {
  background-color: #FEE;
  border: 1px solid #EDD;
  color: #A66;
}
</style>

<details >
  <summary style="font-size: 18px;font-weight: bolder">Installing CGP from PyPi</summary>
<div style="border: #00B0F0 solid 3px;border-radius: 10px">
  <div class="alert error">
    <span class="alertClose">X</span>
    <span class="alertText">Uh Oh! We haven't implemented this option yet.
    <br class="clear"/></span>
  </div>
  </div>
</details>

<details >
  <summary style="font-size: 18px;font-weight: bolder">Installing CGP from PyPi</summary>
<div style="border: #00B0F0 solid 3px;border-radius: 10px">
  <div class="alert error">
    <span class="alertClose">X</span>
    <span class="alertText">Uh Oh! We haven't implemented this option yet.
    <br class="clear"/></span>
  </div>
  </div>
</details>

<details >
  <summary style="font-size: 18px;font-weight: bolder">Installing CGP from PyPi</summary>
<div style="border: #00B0F0 solid 3px;border-radius: 10px">
  <div class="alert error">
    <span class="alertClose">X</span>
    <span class="alertText">Uh Oh! We haven't implemented this option yet.
    <br class="clear"/></span>
  </div>
</div>
</details>
<details>
  <summary style="font-size: 18px;font-weight: bolder">Installing CGP from Source</summary>
<div style="border: #00B0F0 solid 3px;border-radius: 10px">
  <p>To install the CGP from source, you'll first need to clone the directory directly from this github page. To do so, simply</p>
  <code>
        >>> pip install git+https://www.github.com/jzuhone/cluster_generator
  </code>
  <p>This will clone the directory directly into your site-packages directory and then run the <code>setup.py</code> installer.</p>
<p>Once installation has completed, you should be able to access the library simply using</p>
<code>
import cluster_generator as cgp
</code>
  <div class="alert notice">
    <span class="alertClose">X</span>
    <span class="alertText" style="text-align:left;font-size:14px; margin-left:0px; margin-right:0px">Once installation has been completed, you should be able to see the installation directory from<br>
<code>pip show cluster_generator</code><br>You can then find the configuration file within the <code>install_dir/bin</code> directory.
    <br class="clear"/></span>
  </div>
</div>
</details>

### Dependencies

`cluster_generator` is compatible with Python 3.8+, and requires the following
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

All contributions, bug fixes, documentation improvements, and ideas are welcome. If you're interested in pursuing further development of the
Cluster Generator Project, we suggest you start by browsing the [API Documentation](https://eliza-diggins.github.io/cluster_generator/build/html/api.html). When you're ready
create a fork of this branch and begin your development. When you finish,
feel free to  add a pull request to this repository and we will review your code contribution.

## Licence
