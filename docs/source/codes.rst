.. _codes:

Simulation Software
-------------------

The hallmark of the CGP is the ability to generate 1 model in ``cluster_generator`` and export it to a variety of simulation codes
with ease. On this page, we've provided a list of the various simulation software we support and some software specific tips on
how to use each of them.

.. _flash:

``FLASH``
=========

|medium-support| |req-ext| |AMR| |documentation_bad|

The ``FLASH`` 3.0 hydrodynamics code [FrOlRi]_ (developed at the `University of Chicago <http://flash.uchicago.edu>`_) is an all purpose simulation
code which relies on adaptive mesh refinement and the piecewise-parabolic method (PPM).

The ``FLASH`` simulation setup ``GalaxyClusterMerger`` can be used with the
inputs to ``cluster_generator`` to perform a single-cluster simulation or a
binary merger. ``GalaxyClusterMerger`` has two modes in which it can be run in:
``RigidGravity``, which uses rigid potentials to represent the dark matter and
stars, or ``SelfGravity``, which represents the dark matter and stars as
particles. ``GalaxyClusterMerger`` also supports the inclusion of a tangled
magnetic field in the simulations.

.. admonition:: Info
    The ``GalaxyClusterMerger`` setup is currently available from John ZuHone by
    request for users who have a licensed copy of the ``FLASH`` code.

``GAMER``
=========

|fully-supported| |AMR| |documentation_bad|

``GAMER`` (GPU-accelerated Adaptive MEsh Refinement code [ScTsCh]_) utilizes a modified AMR approach designed
to utilize the acceleration potential of GPUs.
The ``GAMER`` simulation setup ``ClusterMerger`` can be used with the inputs to
``cluster_generator`` to perform a single-cluster simulation, binary merger, or
a triple merger. The required, optional, and recommended ``Makefile`` options
for the ``ClusterMerger`` setup are as follows:

* ``SIMU_OPTION += -DMODEL=HYDRO`` (required)
* ``SIMU_OPTION += -DGRAVITY`` (required)
* ``SIMU_OPTION += -DPARTICLE`` (required)
* ``SIMU_OPTION += -DNCOMP_PASSIVE_USER=n`` (required, n = ``Merger_Coll_NumHalos+Merger_Coll_UseMetals``, see below )
* ``SIMU_OPTION += -DMHD`` (optional, for simulations with magnetic fields)
* ``SIMU_OPTION += -DEOS=EOS_GAMMA`` (required)
* ``SIMU_OPTION += -DPAR_NATT_USER=2`` (required, supports the type and halo attributes)
* ``SIMU_OPTION += -DGPU`` (recommended)
* ``SIMU_OPTION += -DLOAD_BALANCE=HILBERT`` (recommended)
* ``SIMU_OPTION += -DOPENMP`` (recommended)
* ``SIMU_OPTION += -DSUPPORT_HDF5`` (required)

``Arepo``
=========

|fully-supported| |SPH| |documentation_easy|

``Arepo`` [WeSpPa]_ is a smooth-particle hydrodynamics simulation software developed at the `Max-Planck Institute for Astrophysics <https://arepo-code.org/wp-content/userguide/index.html>`_.
CGP can be used to create HDF5-based initial conditions for
one, two, or three clusters for an idealized merger simulation for the Arepo code.
The way to do this is to create a Gadget-like HDF5 initial conditions file:

After generating the initial conditions file, the ICs will need to be processed
through two steps before the simulation proper can be run. The first step is to
add a background grid of cells to the particle distribution. This is done with
the ``ADD_BACKGROUND_GRID`` Config option in Arepo. The Arepo configuration
options to be enabled in ``Config.sh`` for this step include:



``GIZMO``
=========

Coming soon!

``Enzo``
========

Coming soon!

``RAMSES``
==========

|fully-supported| |req-ext| |AMR| |documentation_medium| |MOND|

The ``RAMSES`` code is a multipurpose AMR software developed by `Romain Teyssier <https://bitbucket.org/rteyssie/ramses/src/master/>`_. Because of its
support for minimally invasive patching, ``RAMSES`` supports a variety of options for non-newtonian gravity, sub-grid physics, and other
adaptations. In particular, ``RAMSES`` is the only code supported by CGP which is capable of performing MONDian simulations.

Due to the variety of options regarding setting up ``RAMSES`` for use with the CGP, interested users should read the guide below for a
full breakdown of what can be done with ``RAMSES`` and ``CGP``.

.. card:: RAMSES

    In this guide, we show you how to run a simulation in RAMSES from CGP models.

    +++

    |intermediate| |20min| |nyi|


``Athena++``
============

Coming soon!

References
++++++++++

.. [FrOlRi] Fryxell, B., Olson, K., Ricker, P., Timmes, F. X., Zingale, M., Lamb, D. Q., ... & Tufo, H. (2000). FLASH: An adaptive mesh hydrodynamics code for modeling astrophysical thermonuclear flashes. The Astrophysical Journal Supplement Series, 131(1), 273.
.. [ScTsCh] Schive, H. Y., Tsai, Y. C., & Chiueh, T. (2010). GAMER: a graphic processing unit accelerated adaptive-mesh-refinement code for astrophysics. The Astrophysical Journal Supplement Series, 186(2), 457.
.. [WeSpPa] Weinberger, R., Springel, V., & Pakmor, R. (2020). The Arepo public code release. The Astrophysical Journal Supplement Series, 248(2), 32.

.. |low-support| image:: https://img.shields.io/badge/Support-minimal-orange
.. |medium-support| image:: https://img.shields.io/badge/Support-moderate-blue
.. |fully-supported| image:: https://img.shields.io/badge/Support-full-green
.. |documentation_easy| image:: https://img.shields.io/badge/Documentation-Complete-green
.. |documentation_medium| image:: https://img.shields.io/badge/Documentation-Partial-blue
.. |documentation_bad| image:: https://img.shields.io/badge/Documentation-None-black
.. |req-ext| image:: https://img.shields.io/badge/Requires_Additional_Software-Yes-black
.. |AMR| image:: https://img.shields.io/badge/Code_Type-AMR-purple
.. |SPH| image:: https://img.shields.io/badge/Code_Type-SPH-cyan
.. |MOND| image:: https://img.shields.io/badge/MONDIAN-purple
.. |beginner| image:: https://img.shields.io/badge/Difficulty-Beginner-green
.. |intermediate| image:: https://img.shields.io/badge/Difficulty-Intermediate-blue
.. |advanced| image:: https://img.shields.io/badge/Difficulty-Advanced-black
.. |10min| image:: https://img.shields.io/badge/10min-blue
.. |20min| image:: https://img.shields.io/badge/20min-blue
.. |30min| image:: https://img.shields.io/badge/30min-blue
.. |40min| image:: https://img.shields.io/badge/40min-blue
.. |60min| image:: https://img.shields.io/badge/60min-blue
.. |feature| image:: https://img.shields.io/badge/Feature-purple
.. |nyi| image::  https://img.shields.io/badge/NotYetImplemented-red
