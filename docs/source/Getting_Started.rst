.. _getting_started:

Quickstart Guide
----------------

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

Introduction
============

Galaxy clusters are the largest gravitationally bound objects in the universe
and the current endpoints of the process of cosmological structure formation.
Clusters are made of dark matter (DM), the hot, magnetized, and X-ray-emitting
gas known as the intracluster medium (ICM), and their constituent galaxies.

For many applications, especially those focusing on the properties of the DM
and ICM, clusters can be approximated as roughly spherically symmetric objects.
This is often done using standard radial profiles for the various physical
properties of the cluster, as well as assuming hydrostatic and virial equilibrium.

These models are especially useful for semi-analytic studies and idealized
N-body / (magneto)hydrodynamic simulations of clusters, whether of single objects
or mergers between two or more. This is the purpose of the ``cluster_generator``
package, which generates equilibrium cluster models from radial profile inputs.
These models can be used to produce distributions of gas, DM, and star particles
in hydrostatic and virial equilibrium for input into simulations. There is also
the ability to create magnetic or velocity fields in three dimensions based on
the properties of the cluster(s) which are modeled. Finally, ``cluster_generator``
can be used to set up initial conditions for single-cluster or merger simulations
for a number of N-body/hydrodynamic codes used in the astrophysics commmunity.

.. _installation:

Getting the Package
===================

The ``cluster_generator`` package can be obtained for python versions 3.8 and up. Installation instructions are provided
below for installation from source code, from ``pip`` and from ``conda``.

From PyPI
+++++++++

.. note::

    This feature is not yet available.


With CONDA
++++++++++

.. note::

    This feature is not yet available.

With PIP
++++++++

.. note::

    This feature is not yet available.

From Source
+++++++++++

To install the library directly from source code, there are two options. If you are using / have installed pip, you can
install directly from the github URL as follows:

- Using your preferred environment (venv, local python installation, etc), call

  .. code-block:: bash

      pip install git+https://www.github.com/eliza-diggins/cluster_generator


  This will install directly from this repository without generating a local clone.

- If you're interested in having a local clone, you can instead do the following

  - First, clone the repository using

    .. code-block:: bash

        git clone https://www.github.com/eliza-diggins/cluster_generator

    .. warning::

        Make sure to navigate to a directory where you want the clone to appear.

    Once the clone has been generated, change your directory so that you are inside the clone and in the same directory as the ``setup.py`` script. Then run the following command:

    .. code-block:: bash

        pip install .

    This will install the local clone to your python installations ``site-packages`` directory. If you want to install the package in place, you can use

    .. code-block:: bash

        pip install -e .

    which will install the package in development mode.

    .. warning::

        If you install the clone in editing mode (``-e``), you will have to be in the install directory to import the library.

To test that you've installed the project, simply run

.. code-block:: bash

    pip show cluster_generator


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
- `dill <https://github.com/uqfoundation/dill>`_ [Serialization]
- `halo <https://github.com/manrajgrover/halo>`_ [Progress Spinners]
- `pandas <https://github.com/pandas-dev/pandas>`_ [Dataset Manipulations]

These will be installed automatically if you use ``pip`` or ``conda`` as detailed below.


Though not required, it may be useful to install `yt <https://yt-project.org>`_
for creation of in-memory datasets from ``cluster_generator`` and/or analysis of
simulations which are created using initial conditions from
``cluster_generator``.

Getting Started
===============


To begin using ``cluster_generator``, first install the library by following the instructions in the :ref:`installation` section.

Once you've installed the package, have a look at the note on unit conventions (:ref:`here<units>`). Once you're familiar with the units,
we recommend you begin on this page by reading through our quickstart guide to get your feet wet.

.. raw:: html

   <hr style="height:2px;background-color:black">

Quickstart Guides
+++++++++++++++++

.. card-carousel:: 2

    .. card:: The Basics
        :link: notebooks/quick_notebooks/quickbook
        :link-type: doc

        **Length**: 10 minutes
        ^^^
        Get a basic understanding of the ``cluster_generator`` library and its capabilities. Suitable for first time users.
        +++
        :doc:`Quickstart Guide: The Basics <notebooks/quick_notebooks/quickbook>`


    .. card:: Your First Simulation
        :link: notebooks/quick_notebooks/quickbook2
        :link-type: doc

        **Length**: 20 minutes
        ^^^
        So you know the basics, but you want to start actually doing your science? Use this guide as a reference for the
        entire process of initializing and simulating a cluster from ``cluster_generator``. This guide uses RAMSES as an
        example.
        +++
        :doc:`Quickstart Guide: Your First Simulation <notebooks/quick_notebooks/quickbook2>`

    .. card:: Getting the Most Out of the CGP
        :link: notebooks/quick_notebooks/quickbook3
        :link-type: doc

        **Length**: 40 minutes
        ^^^
        Starting to feel like you know the ropes? This guide will give you some deeper insight into the functionality of the
        library and teach you to use advanced resources like alternative virialization methods, saving profiles, and accessing
        collections.
        +++
        :doc:`Quickstart Guide: Getting the Most Out of the CGP <notebooks/quick_notebooks/quickbook3>`

    .. card:: Getting Funky: MONDian Gravity
        :link: notebooks/quick_notebooks/quickbook4
        :link-type: doc

        **Length**: 20 minutes
        ^^^
        Feeling ready to jump down a rabbit hole? This guide will show you how to use the MONDian gravity theories built
        into cluster generator!
        +++
        :doc:`Quickstart Guide: Getting Funky - MONDian Gravity <notebooks/quick_notebooks/quickbook4>`



If you're already somewhat familiar with
the code and you're looking for something a little more in depth, the following sections may provide a more concrete introduction to
many of the core aspects of the code:

The best place to start is :ref:`radial_profiles`, to see which analytical radial profile models for gas, DM,
and star properties can be used to create equilibrium models, which are discussed
in the :ref:`cluster_models` section. This section begins with some helpful mathematical
background in :ref:`math_overview_models`. For some, this may be all you need.

However, if you want to use these models to generate particles for simulations or
other analysis, check out the :ref:`particles` section. For setting up DM and/or
star particles in virial equilibrium, :ref:`math_overview_particles` provides a
mathematical overview.

If you want to create 3-dimensional magnetic or velocity fields based on the
cluster properties, check out :ref:`fields`.

Finally, though one can follow all of the steps individually to create initial
conditions for simulations, ``cluster_generator`` provides some handy tools and
documentation to create initial conditions for a number of popular codes in the
:ref:`initial_conditions` and :ref:`codes` sections.

Code examples are given throughout the text, but some fully-worked examples of
generating models from start to finish are given in the :ref:`examples` section.


.. _units:

Units
+++++

The unit system assumed in ``cluster_generator`` is designed to use units
appropriate for cluster scales:

* Length: :math:`{\rm kpc}`
* Time: :math:`{\rm Myr}`
* Mass: :math:`{\rm M_\odot}`

From these three, the units for other quantities, such as density, pressure,
specific energy, gravitational potential, etc. are straighforwardly derived.
These are the units which will be assumed for all inputs (whether arrays or
scalars) to the functions in ``cluster_generator``, unless otherwise specified
in the documentation and/or docstrings. What this means in practice is that if
one supplies an input without units, it is assumed to be in the above units
depending on the type of input (e.g., position, velocity, etc.). If you supply
an input with units attached from `yt <https://yt-project.org>`_ or
`unyt <http://unyt.readthedocs.org>`_, it will be converted to the above units
internally before performing any calculations.

Some examples:

Most quantities which are returned as outputs of ``cluster_generator`` functions
have units attached, using `unyt <http://unyt.readthedocs.org>`_. These are usually in the above unit system.
For some output quantities, these units are sometimes used:

* Number density: :math:`{\rm cm^{-3}}`
* Temperature: :math:`{\rm keV}`
* Entropy: :math:`{\rm keV~cm^2}`

.. raw:: html

   <hr style="height:10px;background-color:black">





