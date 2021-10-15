.. _codes:

Setting Up Idealized Cluster Problems in Various Codes
------------------------------------------------------

``FLASH``
=========

The ``FLASH`` simulation setup ``GalaxyClusterMerger'' can be used with the 
inputs to ``cluster_generator'' to perform a single-cluster simulation or a 
binary merger. ``GalaxyClusterMerger'' has two modes in which it can be run in: 
``RigidGravity'', which uses rigid potentials to represent the dark matter, or
``SelfGravity'', which represents the dark matter as particles. 

``GAMER``
=========

The GAMER simulation 

Arepo
=====

``cluster_generator'' can be used to create HDF5-based initial conditions for
one, two, or three clusters for an idealized merger simulation. 

After generating the initial conditions file, the ICs will need to be processed
through two steps before the simulation proper can be run. The first step is to
add a background grid of cells to the particle distribution. This is done with
the ``ADD_BACKGROUND_GRID'' Config option in AREPO. 

GIZMO
=====

Coming soon!

Enzo
====

Coming soon!

Ramses
======

Coming soon!

Athena++
========

Coming soon!
