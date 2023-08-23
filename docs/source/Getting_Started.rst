.. _quickstart:

Quickstart Guide
----------------
This quickstart guide will provide the basic tools necessary to successfully produce the initial conditions for whichever
code you choose to use for your simulations. Below, you'll find instructions on installing ``cluster_generator``.

.. raw:: html

   <hr style="height:10px;background-color:black">

.. _installation:

Getting the Package
===================

From PyPI
+++++++++
.. attention::

    This package is not yet published on PyPI

From Source
+++++++++++
To gather the necessary code from source, simple navigate to a directory in which you'd like to store the local copy
of the package and execute

.. code-block:: bash

    git clone https://github.com/jzuhone/cluster_generator

If you want a specific branch of the project, use the ``-b`` flag in the command and provide the name of the branch.

Once the git clone has finished, there should be a directory ``./cluster_generator`` in your current working directory.

.. raw:: html

   <hr style="height:3px;background-color:black">


Dependencies
++++++++++++

``cluster_generator`` is compatible with Python 3.8+, and requires the following
Python packages:

- `unyt <http://unyt.readthedocs.org>`_ [Units and quantity manipulations]
- `numpy <http://www.numpy.org>`_ [Numerical operations]
- `scipy <http://www.scipy.org>`_ [Interpolation and curve fitting]
- `h5py <http://www.h5py.org>`_ [h5 file interaction]
- `tqdm <https://tqdm.github.io>`_ [Progress bars]
- `ruamel.yaml <https://yaml.readthedocs.io>`_ [yaml support]

These will be installed automatically if you use ``pip`` or ``conda`` as detailed below.

.. admonition:: Recommended

    Though not required, it may be useful to install `yt <https://yt-project.org>`_
    for creation of in-memory datasets from ``cluster_generator`` and/or analysis of
    simulations which are created using initial conditions from
    ``cluster_generator``.

Installation
++++++++++++

``cluster_generator`` can be installed in a few different ways. The simplest way
is via the conda package if you have the 
`Anaconda Python Distribution <https://store.continuum.io/cshop/anaconda/>`_:

.. code-block:: bash

    [~]$ conda install -c jzuhone cluster_generator

This will install all of the necessary dependencies.

The second way to install ``cluster_generator`` is via pip. pip will attempt to 
download the dependencies and install them, if they are not already installed 
in your Python distribution:

.. code-block:: bash

    [~]$ pip install cluster_generator

Alternatively, to install into your Python distribution from 
`source <http://github.com/jzuhone/cluster_generator>`_:

.. code-block:: bash
    
    [~]$ git clone https://github.com/jzuhone/cluster_generator
    [~]$ cd cluster_generator
    [~]$ python -m pip install .

Getting Started
---------------
Now that you have ``cluster_generator`` installed, we're ready to start using the software. The guides below contain all of the
info new users need to get started using ``cluster_generator``'s API.

.. toctree::
    :maxdepth: 2
    :caption: Guides:

    quick_notebooks/book_1
    quick_notebooks/book_2
    quick_notebooks/book_3

