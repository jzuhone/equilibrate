|yt-project|

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://eliza-diggins.github.io/cluster_generator/build/html/index.html

.. |testing| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg
.. |Pylint| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg
.. |Github Page| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg
.. |coverage| image:: https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg?branch=MOND
   :target: https://coveralls.io/github/Eliza-Diggins/cluster_generator?branch=MOND

|ytproject| |docs| |testing| |Github Page| |Pylint| |coverage|


``cluster_generator`` is a cross-platform galaxy cluster initializer for N-body / hydrodynamics codes. ``cluster_generator`` supports
a variety of different possible configurations for the initialized galaxy clusters, including a variety of profiles, different construction
assumptions, and non-Newtonian gravity options.

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

