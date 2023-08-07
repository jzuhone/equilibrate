.. _introduction:

Introduction
------------

Galaxy clusters are the largest gravitationally bound objects in the universe
and the current endpoints of the process of cosmological structure formation. 
Clusters are made of dark matter (DM), the hot, magnetized, and X-ray-emitting 
gas known as the intracluster medium (ICM), and their constituent galaxies. 

For many applications, especially those focusing on the properties of the DM
and ICM, clusters can be approximated as roughly spherically symmetric objects.
This is often done using standard radial profiles for the various physical
properties of the cluster, as well as assuming hydrostatic and virial equilibrium. 

These models are especially useful for semi-analytic studies and idealized 
N-body/(magneto)hydrodynamic simulations of clusters, whether of single objects
or mergers between two or more. This is the purpose of the ``cluster_generator``
package, which generates equilibrium cluster models from radial profile inputs. 
These models can be used to produce distributions of gas, DM, and star particles
in hydrostatic and virial equilibrium for input into simulations. There is also
the ability to create magnetic or velocity fields in three dimensions based on
the properties of the cluster(s) which are modeled. Finally, ``cluster_generator``
can be used to set up initial conditions for single-cluster or merger simulations
for a number of N-body/hydrodynamic codes used in the astrophysics commmunity.

Getting Started
===============

After reading the note on :ref:`units` below, the best place to start is
:ref:`radial_profiles`, to see which analytical radial profile models for gas, DM, 
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
=====

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