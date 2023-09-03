.. _cluster_models:

Cluster Models
--------------
The :py:class:`model.ClusterModel` class is one of the core structures available in ``cluster_generator``. These objects hold all of
the data related to a single cluster being modeled. The user can use a :py:class:`model.ClusterModel` object to generate particle distributions,
enforce hydrostatic equilibrium, virialize halos, and generate intial conditions. Furthermore, the user can generate
:py:class:`model.ClusterModel` objects using a variety of pre-built protocols using :py:class:`radial_profiles.RadialProfile` objects to provide necessary fields of data.

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

.. _math_overview_models:

Cluster Models: Mathematical Overview
=====================================
.. admonition:: Reader Suggestion

    To learn more about the various gravity theories discussed here, see the :ref:`gravity` page.

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

    \nu\left(\frac{|\nabla \Psi |}{a_0}\right)\nabla \Psi = \Gamma,\;\;\nabla \Psi = \frac{GM_{\mathrm{dyn}}}{r^2}

This form is not analytically solvable for general :math:`\mu`; however, implicit solutions can be found numerically which provide the
dynamical mass distribution. In the DMR, :math:`\nu(x) \to x^{-1/2}`, so

.. math::

    (a_0\nabla \Psi)^{1/2} = \Gamma \implies M_{\mathrm{dyn}} \sim r^2 \Gamma^2 \sim r^{2\beta},

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
| QUMOND       | :math:`\Gamma = \nu\left(\frac{|\nabla \Psi|}{a_0}\right) \nabla \Psi` where      | :math:`T\sim r^{0}`        |
|              | :math:`\nabla \Psi = GM_{\mathrm{dyn}}(<r)/r^2`                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+

.. raw:: html

   <hr style="height:10px;background-color:black">

Generating a :py:class:`model.ClusterModel` Using Radial Profiles
=================================================================

Based on the mathematics above, there are a variety of ways to produce :py:class:`model.ClusterModel` objects. Most of the common approaches
that see use in practice are built into the ``cluster_generator`` package; however, the :py:meth:`~model.ClusterModel.from_arrays` class method can
be used to generate a :py:class:`model.ClusterModel` object manually. The available generation approaches are listed as follows:

+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+
| Method                          |                                 Function                                 | Description                                                      |
+=================================+==========================================================================+==================================================================+
| From :math:`\rho_g`             | :py:meth:`~model.ClusterModel.from_dens_and_tden`                        | Generates the galaxy cluster from the gas and dynamical density  |
| and :math:`\rho_{\mathrm{dyn}}` |                                                                          | profiles. Computes temperature / grav. field.                    |
+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+
| From :math:`\rho_g`             | :py:meth:`~model.ClusterModel.from_dens_and_temp`                        | Generates the galaxy cluster from the gas density and temperature|
| and :math:`T_g`                 |                                                                          | profiles. Computes total mass, dm, stellar etc.                  |
+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+
| From :math:`\rho_g`             |  :py:meth:`~model.ClusterModel.from_dens_and_entr`                       | Generates the galaxy cluster from the gas density and entropy    |
| and :math:`\rho_{\mathrm{dyn}}` |                                                                          | profiles. Computes total mass, dm, stellar etc.                  |
+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+

``ClusterModel`` from Gas Density and Gas Temperature Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The user may generate a :py:class:`model.ClusterModel` from the :py:meth:`model.ClusterModel.from_dens_and_temp` method, which requires :py:class:`radial_profiles.RadialProfile` objects
for the temperature and the gas density. Using the condition of hydrostatic equilibrium, the potential, pressure, dynamical mass, and other
necessary fields are automatically computed based on the chosen gravity theory.

The pressure :math:`P(r)` is provided via the ideal gas law:

.. math::

    P(r) = \frac{\rho_g(r) k_b T(r)}{m_p \eta},

where :math:`\eta` is the mean-molecular mass (generally 0.6 for galaxy clusters).

Once the pressure is determined, Euler's Equations can be used for an incompressible fluid, yielding

.. math::

    \Gamma = \frac{-\nabla P(r)}{\rho_g} = \nabla \Phi

Once the potential has been determined, the corresponding gravity theory is applied to determine the dynamical mass of the system and
by extension determine the necessary halo component.

.. math::

    M_{dm} = M_{\mathrm{dyn}} - M_{\mathrm{bary}}.

+-----------------------+-----------------------------------+
|Provided               | Computed                          |
+=======================+===================================+
| :math:`T_g`           | :math:`\rho_{dm}, M_{dm}, M_{dyn}`|
| :math:`\rho_{g}`      | :math:`S, \Phi, \nabla \Phi, P`   |
| :math:`\rho_{\star}`  |                                   |
+-----------------------+-----------------------------------+



``ClusterModel`` from Gas Density and Total Density Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Users may also generate a :py:class:`model.ClusterModel` object from an entropy profile instead of the corresponding temperature profile.
There is a 1-to-1 correspondence between entropy and temperature given that the gas density is fixed, therefore, the procedure
follows exactly from the previous section; however, the entropy is converted first to a temperature profile using the widely accepted formula

.. math::

    S(r) = k_b T_g(r)n_e(r)^{-2/3}.


``ClusterModel`` Without Gas
++++++++++++++++++++++++++++++++++++++++++
Not all of the :py:class:`model.ClusterModel` objects are generated with a gas density profile. If one wants to produce a gas-less model,
the :py:meth:`model.ClusterModel.no_gas` method is available. Users must provide a dynamical density profile, and are optionally allowed
to provide a stellar component. See the API reference for syntactical information.

Checking the Hydrostatic Equilibrium
====================================
In the vast majority of cases, the aim of the :py:class:`model.ClusterModel` object is to produce a model which is in hydrostatic equilibrium. To check how successful the algorithm was in
generating a hydrostatic galaxy cluster, one can use the :py:meth:`model.ClusterModel.check_hse` method, which will provide the maximal relative deviation of the system from hydrostatic equilibrium.

This is done by recalling that the hydrostatic equilibrium condition requires that

.. math::

    -\frac{\nabla P}{\rho_g} = \nabla \Phi.

This, we define the hydrostatic variable :math:`\xi` such that

.. math::

    \xi = \frac{\nabla \Phi \rho_g + \nabla \Phi}{\nabla \Phi \rho_g}.

clearly, :math:`\xi \approx 0` indicates a successfully equilibrated cluster.

Setting a Magnetic Field Strength Profile
=========================================
Magnetic fields play a major role in the dynamics of the ionized plasma modeled in hydro simulations of galaxy clusters. In ``cluster_generator``, there
are currently two approaches for generating a magnetic field for a galaxy cluster.

Setting a Magnetic Field from :math:`\beta`
+++++++++++++++++++++++++++++++++++++++++++
Users can specify the magnetic field profile using the :py:meth:`~model.ClusterModel.set_magnetic_field_from_beta`, which allows the user
to specify the magnetic pressure factor :math:`\beta`, which is defined such that

.. math::

    \beta = \frac{p_{\mathrm{thermal}}}{p_{\mathrm{magnetic}}}.

Setting a Magnetic Field from plasma density
++++++++++++++++++++++++++++++++++++++++++++
Another common approach for initializing magnetic fields is to let the magnetic field be proportional to a power of the gas density. Most commonly, this value is
:math:`\eta = \frac{2}{3}`; however, the user may specify whichever exponent they choose. To initialize a cluster with a magnetic field using this approach, use the :py:meth:`~model.ClusterModel.set_magnetic_field_from_density`.

Adding Other Fields
===================
Users may add additional fields to the :py:class`~model.ClusterModel` instance using the :py:meth:`~model.ClusterModel.set_field` method, which takes a
``name`` and ``value`` and loads the corresponding field into the model.

.. attention::

    User's should be aware that ``value``'s being attributed as fields must be ``unyt_array``'s, not generic ``list`` types or ``np.ndarray``. Furthermore,
    the length of the value being specified must match ``ClusterModel.num_elements``.

Reading and Writing :py:class:`model.ClusterModel` Objects to and from Disk
===========================================================================
.. admonition:: User Advice

    Generating :py:class:`model.ClusterModel` instances can be **slow**. As many cases as possible, it is worthwhile to
    write models to disk in one of the formats discussed below so that they can be read more easily.

Many different protocols for writing and reading :py:class:`model.ClusterModel` instances to / from disk have been implemented. The following table summarizes the
options available; however, we advice consulting the API reference for each of the methods to get more information about specifics regarding each of the available formats.

+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+
| Format        | Write Method                                           | Read Method                                         | Notes                             |
+===============+========================================================+=====================================================+===================================+
| ASCII         | :py:meth:`~model.ClusterModel.write_model_to_ascii`    | **Not Yet Implemented**                             | None                              |
+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+
| HDF5          | :py:meth:`~model.ClusterModel.write_model_to_h5`       | :py:meth:`~model.ClusterModel.from_h5_file`         | None                              |
+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+
| BINARY        | :py:meth:`~model.ClusterModel.write_model_to_binary`   | **Not Yet Implemented**                             | None                              |
+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+

Additional Notes
................
.. attention::

    In general, this approach will yield a dark matter halo component regardless of the chosen gravity theory. As such,
    if the user is studying a MOND theory outside of the context of MOND + DM, this approach will not generally be viable
    unless it has already been confirmed that the dynamical mass determined by the hydrostatic equilibrium condition matches
    that contributed by :math:`\rho_\star` and :math:`\rho_g`.

Non-Physical Systems
====================
In many situations, a galaxy cluster may be **mathematically** constructed using pathological, incomplete, or ill-determined profiles which
create (in the case of generation from density and temperature) similarly pathological values for the resulting fields. Because the aim of ``cluster_generator`` is to
reasonably produce realizable galaxy cluster initial conditions, these pathological cases can interfere with the underlying mathematics used in initial condition generation
from determining particle velocities to sampling profiles. Nonetheless, ``cluster_generator`` will not prevent the user from generating a :py:class:`model.ClusterModel` which is non-physical.
Such :py:class:`model.ClusterModel` objects are most easily generated from temperature and density profiles, as their behavior at large and small radii has immediate implications for the
dynamical variables which are derived during generation.

As an example, consider the following :py:class:`model.ClusterModel` object:

.. highlight:: python
.. code-block:: python
    :force:
    :linenos:

    from cluster_generator.model import ClusterModel
    from cluster_generator.radial_profiles import find_overdensity_radius, \
        snfw_density_profile, snfw_total_mass, vikhlinin_density_profile, vikhlinin_temperature_profile, \
        rescale_profile_by_mass, find_radius_mass, snfw_mass_profile

    # Configuring pathological model
    #------------------------------------
    # - Parameters
    z = 0.1
    M200 = 1.5e15
    conc = 4.0

    # - Constructing the density profile
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)
    f_g = 0.12
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g * M500, r500)

    # - Constructing the temperature profile
    temp = vikhlinin_temperature_profile(2.42,-0.02,5.00,1.1,350,1,19,2)
    rhos = 0.02 * rhot
    rmin = 0.1
    rmax = 10000.0

    # - Producing the model
    m = ClusterModel.from_dens_and_temp(rmin, rmax, rhog, temp,
                                        stellar_density=rhos)
    m.set_magnetic_field_from_beta(100.0, gaussian=True)

.. figure:: _images/model/non-physical-example.png

    The resulting profiles after the :py:class:`model.ClusterModel` object has been generated. Notice that
    even thought the gas density and temperature profiles are reasonable, the dynamical mass and
    halo mass have non-physical behaviors.

**What does cluster generator do about this?**

The short answer is **nothing**. At least, not until the user says something. All :py:class:`model.ClusterModel` instances have the
:py:meth:`model.ClusterModel.is_physical` method, which will determine if the system is physical and to what radii.

.. code-block:: python

    m.is_physical()

will yield.

.. code-block:: shell

    ✖ [ClusterModel object; gravity=Newtonian] is non-physical over 35.2% of domain.

 If the user wants to fix the system, they can then call the :py:meth:`model.ClusterModel.rebuild_physical` method, which returns a
newly constructed equivalent of the original model which is physically self-consistent.

.. code-block:: python

    m2 = m.rebuild_physical()
    m2.is_physical()

which will instead yield the much more pleasant

.. code-block:: shell

    ✔ ClusterModel object; gravity=Newtonian is physical.

To do this, all of the density profiles are prevented from reaching 0 and then each field is recomputed with that
new, physically consistent, set of profiles.


.. figure:: _images/model/non-physical-fixed.png

    The result of setting ``require_physical`` during model generation. Here **magenta** is the original model without
    any restriction of physical viability. The profiles for halo (dm) density, and dynamical (total) density clearly have
    regions of non-physical behavior. The **forest-green** corresponds to ``require_physical == True``, and thus removes all
    of the non-physical regions of the density and mass profiles; however, leaves the temperature profile the same. Finally,
    **teal** corresponds to ``require_physical == "rebuild"``, which entirely recomputes the temperature profile after fixing non-physical regions.
