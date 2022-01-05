.. _cluster_models:

Cluster Models
--------------

.. _math_overview_models:

Cluster Models: Mathematical Overview
=====================================

Assuming the intracluster medium of galaxy clusters can be modeled as an
ideal fluid, the momentum density :math:`\rho{\bf v}` of the
gas obeys the Euler momentum equation (here written in conservative form
and ignoring magnetic fields, viscosity, etc.):

.. math::

    \frac{\partial({\rho_g{\bf v}})}{\partial{t}} + \nabla \cdot (\rho_g{\bf v}{\bf v})
    = -\nabla{P} + \rho_g{\bf g}

where :math:`\rho_g` is the gas density, :math:`{\bf v}` is the gas velocity,
:math:`P` is the gas pressure, and :math:`{\bf g}` is the gravitational
acceleration. The assumption of hydrostatic equilibrium implies that 
:math:`{\bf v} = 0` and that all time derivatives are zero, giving:

.. math::

    \nabla{P} = \rho_g{\bf g}

Under the assumption of spherical symmetry, all quantities are a function
of the scalar radius :math:`r` only, :math:`{\bf g} = -GM(<r)\hat{{\bf r}}/r^2`, 
and this simplifies to:

.. math::

    \frac{dP}{dr} = -\rho_g(r)\frac{GM(<r)}{r^2}

where :math:`G` is the Newtonian gravitational constant, and :math:`M(<r)` is 
the total enclosed mass of all components (dark and baryonic matter) within 
:math:`r`. 

The gravitational potential :math:`\phi(r)` is defined by:

.. math::

    \phi(r) = -\frac{GM(<r)}{r} - 4\pi{G}\int_r^\infty{\rho(r)rdr}

where :math:`\rho(r)` is the total mass density of all components. 

Generating a ``ClusterModel`` Using Radial Profiles
===================================================

The above equations can be solved for in a number of ways, 
depending on what the initial assumptions are. 

``ClusterModel`` from Gas Density and Gas Temperature Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ClusterModel`` from Gas Density and Total Density Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ClusterModel`` from Gas Density and Gas Entropy Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

``ClusterModel`` Without Gas
++++++++++++++++++++++++++++

If you want to model a dark matter halo without gas (but potentially with stars), 
set up a total density profile using the 
:meth:`~cluster_generator.cluster_model.Cluster_model.no_gas` method:

.. code-block:: python

    import cluster_generator as cg
    # Create a Hernquist total density profile
    M_0 = 5e14 # units of Msun
    a = 500.0 # units of kpc
    total_density = cg.hernquist_density_profile(M_0, a)
    # Create the virial model
    rmin = 0.1 # minimum radius in kpc
    rmax = 10000.0 # maximum radius in kpc
    num_points = 1000 # the number of samples along the radial profile, optional
    p = cg.ClusterModel.no_gas(rmin, rmax, total_density, 
                               num_points=num_points)
                                            
If there are stars in the cluster model (and not dark matter only) then it is 
possible to supply a stellar mass density profile as well via the 
``stellar_density`` argument:

.. code-block:: python
    
    import cluster_generator as cg
    # Create a Hernquist total density profile
    M_0 = 5e14 # units of Msun
    a = 500.0 # units of kpc
    total_density = cg.hernquist_density_profile(M_0, a)
    # Create the virial model
    rmin = 0.1 # minimum radius in kpc
    rmax = 10000.0 # maximum radius in kpc
    num_points = 1000 # the number of samples along the radial profile, optional    
    # Create a Hernquist stellar density profile
    M_star = 5.0e12 # units of Msun
    a_star = 50.0 # units of kpc
    stellar_density = cg.hernquist_density_profile(M_star, a_star)
    p = cg.ClusterModel.no_gas(rmin, rmax, total_density, 
                               stellar_density=stellar_density)

Checking the Hydrostatic Equilibrium
====================================


Setting a Magnetic Field Strength Profile
=========================================

Adding Other Fields
===================

Reading and Writing ``ClusterModel`` Objects to and from Disk
=============================================================


