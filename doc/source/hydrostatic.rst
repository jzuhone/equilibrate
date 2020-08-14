.. _hydrostatic:

Generating Hydrostatic Equilibrium Profiles
-------------------------------------------

Overview
========

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

    \frac{dP}{dr} = -\rho_g\frac{GM(<r)}{r^2}

where :math:`G` is the Newtonian gravitational constant, and :math:`M(<r)` is 
the total enclosed mass of all components (dark and baryonic matter) within 
:math:`r`. 

This equation can be solved in a number of ways, depending on what the initial
assumptions are. 