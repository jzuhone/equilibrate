
Installing ``cluster_generator``
--------------------------------

Dependencies
============

``cluster_generator`` is compatible with Python 3.8+, and requires the following
Python packages:

- `unyt <http://unyt.readthedocs.org>`_
- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `h5py <http://www.h5py.org>`_
- `tqdm <https://tqdm.github.io>`_
- `ruamel.yaml <https://yaml.readthedocs.io>`_

These will be installed automatically if you use ``pip`` or ``conda`` as detailed below.

Though not required, it may be useful to install `yt <https://yt-project.org>`_ 
for creation of in-memory datasets from ``cluster_generator`` and/or analysis of 
simulations which are created using initial conditions from 
``cluster_generator``.

Installation
============

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

