.. _virial:

Generating Virial Equilibrium Models
------------------------------------

The dark matter and stellar components of galaxy clusters in 
``cluster_generator`` are modeled assuming collisionless dynamics in
virial equilibrium. For collisionless particles with speeds :math:`v` 
positioned at radii :math:`r` from the center of the cluster, the 
gravitational potential is :math:`\Phi(r)`, and the energy distribution 
of particles is given from Eddington:

.. math::

    f({\cal E}) = \frac{1}{\sqrt{8}\pi^2}\left[\int^{\cal E}_0{d^2\rho \over d\Psi^2}{d\Psi
    \over \sqrt{{\cal E} - \Psi}} + \frac{1}{\sqrt{{\cal E}}}\left({d\rho \over d\Psi}\right)_{\Psi=0} \right]

where :math:`{\cal E} = \Psi - \frac{1}{2}v^2` and :math:`\Psi = -\Phi`. 
For our equilibrium models, this equation must (in general) be solved 
numerically, even if the underlying dark matter, stellar, and gas densities 
can be expressed analytically. 

The integration of the Eddington formula is handled in ``cluster_generator``
using the :class:`~cluster_generator.virial.VirialEquilibrium` class. Virial
equilibria are rather simple configurations, so this class only takes a few
options.

If you want to model a dark matter halo by itself, 