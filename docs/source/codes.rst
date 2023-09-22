.. _codes:

Setting Up Idealized Cluster Problems in Various Codes
------------------------------------------------------

.. _flash:

``FLASH``
=========

The ``FLASH`` simulation setup ``GalaxyClusterMerger`` can be used with the 
inputs to ``cluster_generator`` to perform a single-cluster simulation or a 
binary merger. ``GalaxyClusterMerger`` has two modes in which it can be run in: 
``RigidGravity``, which uses rigid potentials to represent the dark matter and
stars, or ``SelfGravity``, which represents the dark matter and stars as 
particles. ``GalaxyClusterMerger`` also supports the inclusion of a tangled
magnetic field in the simulations. 

The ``GalaxyClusterMerger`` setup is currently available from John ZuHone by
request for users who have a licensed copy of the ``FLASH`` code. 

.. _gamer:
``GAMER``
=========

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

.. _arepo:

Arepo
=====

``cluster_generator'' can be used to create HDF5-based initial conditions for
one, two, or three clusters for an idealized merger simulation for the Arepo code.
The way to do this is to create a Gadget-like HDF5 initial conditions file:



After generating the initial conditions file, the ICs will need to be processed
through two steps before the simulation proper can be run. The first step is to
add a background grid of cells to the particle distribution. This is done with
the ``ADD_BACKGROUND_GRID`` Config option in Arepo. The Arepo configuration 
options to be enabled in ``Config.sh`` for this step include:


.. _gizmo:

GIZMO
=====

Coming soon!

.. _enzo:

Enzo
====

Coming soon!

.. _ramses:

Ramses
======

RAMSES is an adaptive mesh refinement (AMR) code for hydrodynamical/N-body simulations. It is capable of running
cosmological simulations, incorporating radiative cooling, AGN feedback, and a host of other features. RAMSES is also capable of performing
simulations in MOND gravity using it's RaYMOND patch. The following section will guide you through the ins and outs of generating
RAMSES initial conditions.


Generating your ICs
-------------------

The first step is to configure your initial conditions in whichever way you please. Once you have a :py:class:`ics.ClusterICs` instance,
you can use the :py:func:`codes.setup_ramses_ics` to generate the necessary files.

.. warning::

    These files are placed directly into the working directory. Please be sure to change your directory to the intended final
    location before generating the files.

You should see a ``MergerConfig.txt`` file in your directory along with a ``halo_*_parts.h5`` and ``halo_*_prof.h5`` for each
of your halos. These contain all of the data that RAMSES needs to read the data.

Configuring RAMSES
------------------

.. note::

    The RAMSES patch discussed in this section is currently unavailable.

Once you have obtained the ``cluster_merger`` patch for RAMSES, the installation process should be the same as any other installation
of RAMSES, except that the ``patch`` line in the ``Makefile`` must be provided with the correct file path of the ``cluster_merger`` patch.

Once installed, RAMSES should be fully functional; however, some changes to the standard namelist are required before simulations can be performed.

Running the Simulation
----------------------



.. _athena:

Athena++
========

Coming soon!
