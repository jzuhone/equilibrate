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

We therefore define the "hydrostatic vector", :math:`\Gamma` of the system to be

.. math::

    \Gamma = -\frac{\nabla P}{\rho_g} = \nabla \Phi.

In the context of galaxy clusters, we may typically assume an ideal gas behavior for the ICM, and therefore,

.. math::

    \Gamma = -\frac{\nabla P}{\rho_g} = \frac{-k_b T}{m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

As such, we recognize that from :math:`\rho_g` and :math:`T`, we can generally obtain the acceleration and therefore the dynamical mass
of the system. Because the dynamical mass need not be equivalent to the gasseous mass, it follows that finding the dynamical mass will also
yield the halo mass of the system.

It is worth noting throughout the remainder of this document that for :math:`r\to \infty`, if :math:`\rho \sim r^\alpha` and :math:`T\sim r^\beta`,
then

.. math::

    \Gamma \sim r^{\beta-1}.

This point becomes important as we begin discussing the interaction between the gas dynamics above and the potential of the system.


It is here that the theory begins to diverge dependent on the theoretical context on which the gravity of your system is
based.

Newtonian Gravity
+++++++++++++++++

If the model relies on Newtonian gravity, then :math:`\nabla \Phi = GM_{\mathrm{dyn}}(<r)/r^2`, and therefore, the
dynamical mass is

.. math::

    M_{\mathrm{dyn}}(<r) = \frac{r^2}{G}\Gamma = \frac{-r^2 k_b T}{G m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right]

Thus, the dynamical mass may be directly obtained and the halo mass found by simply removing the known baryonic component. It should be noted that
for :math:`M_{\mathrm{dyn}}(<r)` to be asymptotically constant for large :math:`r`,

.. math::

    M_{\mathrm{dyn}} \sim r^2\Gamma \sim r^{\beta + 1} = r^0,

Therefore, :math:`T \sim 1/r` for a stable dynamical mass profile at large radii.

MOND Gravity: AQUAL
+++++++++++++++++++
In the case of AQUAL, the mass and the acceleration are instead related by the modified Poisson equation, which implies that

.. math::

    M_{\mathrm{dyn}} = \frac{r^2}{G}\mu\left(\frac{| \Gamma |}{a_0}\right)\Gamma.

Similar to the case in Newtonian dynamics, this can simply be taken directly to obtain the correct form of the dynamical mass; however, it should
be noted that the asymptotic stability conditions are not the same as in Newtonian gravity. Specifically, in the deep-MOND regime, where :math:`\mu(x) \approx x`,

.. math::

    M_{\mathrm{dyn}} \approx \frac{r^2}{G}\frac{\Gamma^2}{a_0} \sim r^2r^{2\beta-2},

and thus :math:`\beta = 0` is required for a stable dynamical mass profile.

MOND Gravity: QUMOND
++++++++++++++++++++
In QUMOND, the situation is more complex. The necessary equation becomes

.. math::

    \mu\left(\frac{|\nabla \Psi |}{a_0}\right)\nabla \Psi = \Gamma,\;\;\nabla \Psi = \frac{GM_{\mathrm{dyn}}}{r^2}

This form is not analytically solvable for general :math:`\mu`; however, implicit solutions can be found numerically which provide the
dynamical mass distribution. Again, in the DMR, the equation simplifies to the form

.. math::

    \frac{G^2M_{\mathrm{dyn}}^2}{r^4 a_0} = \Gamma \implies M_{\mathrm{dyn}} \sim r^2 \sqrt{\Gamma} \sim \sqrt{r{3+\beta}},

Thus, :math:`T(r) \sim r^{-3}` at large radii for a stable dynamical mass profile. These results are summarized in the table below.

+--------------+-----------------------------------------------------------------------------------+----------------------------+
| Gravity Type | :math:`\Gamma` and :math:`\Phi` relationship                                      | Mass Stability Condition   |
+==============+===================================================================================+============================+
| Newtonian    | :math:`\Gamma = \nabla \Phi = GM_{\mathrm{dyn}}(<r)/r^2`                          | :math:`T\sim r^{-1}`       |
|              |                                                                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+
| AQUAL        | :math:`\mu\left(\frac{|\Gamma|}{a_0}\right)\Gamma = GM_{\mathrm{dyn}}(<r)/r^2`    | :math:`T\sim r^{0}`        |
|              |                                                                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+
| QUMOND       | :math:`\Gamma = \mu\left(\frac{|\nabla \Psi|}{a_0}\right) \nabla \Psi` where      | :math:`T\sim r^{-3}`       |
|              | :math:`\nabla \Psi = GM_{\mathrm{dyn}}(<r)/r^2`                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+

Dealing With Non-Convergent Mass Profiles
+++++++++++++++++++++++++++++++++++++++++
It is entirely possible to consider clusters wherein :math:`T` does not behave in the correct way at large radii to be
assymptotically consistent with the chosen gravity theory. In cases where :math:`\beta < \beta_0` (where :math:`\beta_0` is the
necessary stable behavior of :math:`T`) the temperature drops off too quickly and
the dynamical mass profile will reach a maxima at some :math:`r_{\mathrm{max}}` before falling to 0. Beyond this maxima, the system is
clearly non-physical and could not possibly be initialized in a simulation.

Similarly, if :math:`\beta > \beta_0`, the dynamical mass will fail to converge and instead increase monotonically.

.. attention::

    To solve this issue, the ``cluster_generator`` package will impose an artificial constraint on the system. At
    :math:`r_{\mathrm{truncate}} = 2r_{200}`, the temperature profile is altered (for the dynamical mass computation only) such that

    .. math::

        T_{\mathrm{truncated}}(r) = T(r) \cdot \left(\frac{r}{r_{\mathrm{truncate}}}\right)^{\beta_0 - \beta}

    Which will induce asymptotic convergence.

.. warning::

    If the user specifies the ``change_T`` kwarg in the ``ClusterModel`` object, then the altered :math:`T` will be
    used as the consistent temperature profile of the entire model, thus saving HSE to a perfect degree, but changing the
    user's intended temperature profile. If ``change_T == False`` (default), the cluster's temperature profile remains the same, but
    the dynamical mass is computed using the altered temperature profile. Thus the user's configuration is saved, but the
    resultant cluster may not be in perfect HSE.



Generating a ``ClusterModel`` Using Radial Profiles
===================================================

The above equations can be solved for in a number of ways, 
depending on what the initial assumptions are. 

``ClusterModel`` from Gas Density and Gas Temperature Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
When the gas density and gas temperature are specified, the condition of hydrostatic equilibrium can be used to
provide the necessary halo. From the Euler momentum equation, :math:`-\nabla P / \rho_g = \Gamma(T,\rho_g) = \nabla \Phi`. As such
the relationship between :math:`\nabla \Phi` and :math:`\Gamma` is exploited to determine the total dynamical mass of the system. From
the dynamical mass profile, the necessary dark matter profile becomes

.. math::

    M_{dm} = M_{\mathrm{dyn}} - M_{\mathrm{bary}}.

``ClusterModel`` from Gas Density and Total Density Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
In the case where the dynamical density is already known, the halo mass function is trivially obtained from

.. math::

    \rho_{\mathrm{halo}} = \rho_{\mathrm{dyn}} - \rho_{\mathrm{bary}},

where the baryonic component may contain a stellar distribution if the user supplies one. The difficult in this case
is to efficiently determine the temperature profile which yields HSE. Recalling that

.. math::

    \Gamma = -\frac{\nabla P}{\rho_g} = \frac{-k_b T}{m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

The differential equation may be inverted, yielding

.. math::

    T(r) = \frac{-m_p \eta}{k_b \rho_g} \int_{r_0}^r \rho_g(r') \nabla \Phi(r') dr' + \frac{\rho_g(r_0)}{\rho_g(r)}T(r_0).

The most efficient approach here is to take :math:`r_0 = \infty`, in which case,

.. math::

    T(r) = \frac{m_p \eta}{k_b \rho_g} \int_{r}^\infty \rho_g(r') \nabla \Phi(r') dr'.

From this, the temperature profile is obtained.

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


