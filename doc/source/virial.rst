.. _virial:

Generating Virial Equilibrium Models
------------------------------------

Overview
========

The dark matter and stellar components of galaxy clusters in 
``cluster_generator`` are modeled assuming collisionless dynamics in
virial equilibrium. The mass density of such a system can be derived 
by integrating the phase-space distribution function 
:math:`f({\bf r}, {\bf v})` over velocity space: 

.. math::

    \rho({\bf r}) = \int{f({\bf r}, {\bf v})d^3{\bf v}}

where :math:`{\bf r}` and :math:`{\bf v}` are the position and velocity
vectors. Assuming spherical symmetry and isotropy, all quantities are 
simply functions of the scalars :math:`r` and :math:`v`, and 
:math:`d^3{\bf v} = 4\pi{v^2}dv`:

.. math::

    \rho(r) = 4\pi\int{f(r, v)v^2dv}

Assuming zero net angular momentum for the cluster, there is a unique 
distribution function :math:`f(E)` which corresponds to the density
:math:`\rho(r)`. Since the total energy of a particle is 
:math:`E = v^2/2 + \Phi` (where :math:`\Phi(r)` is the gravitational
potential) and further defining :math:`\Psi = -\Phi` and 
:math:`{\cal E} = -E = \Psi - \frac{1}{2}v^2`, we find:

.. math::

    \rho(r) = 4\pi\int_0^{\Psi}f({\cal E})\sqrt{2(\Psi-{\cal E})}d{\cal E}

After differentiating this equation once with respect to :math:`\Psi` and
inverting the resulting Abel integrel equation, we finally have:

.. math::

    f({\cal E}) = \frac{1}{\sqrt{8}\pi^2}\left[\int^{\cal E}_0{d^2\rho \over d\Psi^2}{d\Psi
    \over \sqrt{{\cal E} - \Psi}} + \frac{1}{\sqrt{{\cal E}}}\left({d\rho \over d\Psi}\right)_{\Psi=0} \right]

which given a density-potential pair for an equilibrium halo, can be used to
determine particle speeds. For our cluster models, this equation must 
(in general) be solved numerically, even if the underlying dark matter, 
stellar, and gas densities can be expressed analytically. 

The :class:`~cluster_generator.virial.VirialEquilibrium` Class
==============================================================

The integration of the Eddington formula is handled in ``cluster_generator``
using the :class:`~cluster_generator.virial.VirialEquilibrium` class. There are
two ways to create a :class:`~cluster_generator.virial.VirialEquilibrium` object.

If you want to model a dark matter halo without gas, set up a total density
profile using the :meth:`~cluster_generator.virial.VirialEquilibrium.from_scratch`
method:

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
    vir = cg.VirialEquilibrium.from_scratch(rmin, rmax, total_density, 
                                            num_points=num_points)
                                            
If there are stars in the cluster model (and not dark matter only) then it is 
possible to supply a stellar mass density profile as well via the 
``stellar_profile`` argument:

.. code-block:: python
    
    # Create a Hernquist stellar density profile
    M_star = 5.0e12 # units of Msun
    a_star = 50.0 # units of kpc
    stellar_density = cg.hernquist_density_profile(M_star, a_star)
    vir = cg.VirialEquilibrium.from_scratch(rmin, rmax, total_density, 
                                            stellar_profile=stellar_density)

By default, the :meth:`~cluster_generator.virial.VirialEquilibrium.from_scratch`
and :meth:`~cluster_generator.virial.VirialEquilibrium.from_hse_model` methods 
return a :class:`~cluster_generator.virial.VirialEquilibrium` object that
can produce dark matter particles. If you want to create an object that can
produce star particles, set ``ptype="stellar"``. You can even create two 
:class:`~cluster_generator.virial.VirialEquilibrium` objects; one for dark 
matter particles, and the other for stars, from the same input model:

.. code-block:: python

    vir_dm = cg.VirialEquilibrium.from_hse_model(hse, ptype="dark_matter")
    vir_star = cg.VirialEquilibrium.from_hse_model(hse, ptype="stellar")

Checking the Accuracy of the Model
++++++++++++++++++++++++++++++++++


Generating Particles
++++++++++++++++++++
