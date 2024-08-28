.. _theory:
===================================
Cluster Generator Theory
===================================

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

Galaxy Clusters
---------------

Galaxy clusters are the largest gravitationally bound structures in the universe, consisting of hundreds to thousands of galaxies,
vast amounts of hot gas, and dark matter. They are key to understanding the large-scale structure of the cosmos. The hot gas within
clusters emits X-rays, and the distribution of galaxies and gas provides insights into the distribution of dark matter. Galaxy clusters
are formed through the merging of smaller groups and clusters, making them excellent laboratories for studying galaxy formation and evolution,
as well as the physics of the intergalactic medium.

Clusters are found at various redshifts, with nearby clusters (low redshift) like the Virgo Cluster having a redshift :math:`z \approx 0.003`
and distant clusters (high redshift) like those discovered by the Planck satellite having redshifts :math:`z>1`. Studying clusters at different redshifts helps astronomers
understand the evolution of the universe over time.

The Intra-Cluster Medium (ICM)
''''''''''''''''''''''''''''''

The intracluster medium (ICM) is a hot, diffuse plasma that fills the space between galaxies in a galaxy cluster.
It consists primarily of ionized hydrogen and helium, along with trace ammounts of heavier elements such as iron, oxygen, and silicon,
which are the products of supernova explosions within the cluster's galaxies. The temperature of the ICM ranges from :math:`10^7` to
:math:`10^8` Kelvin, making it one of the hottest environments in the universe. The dominant emission is in the X-ray band,
which can be observed with space-based telescopes. The ICM also has interesting electromagnetic behavior in the radio (due to the ICM's magnetic fields).

The dynamics of the ICM are influenced by the gravitational potential of the galaxy cluster, as well as by interactions
between galaxies and the ICM itself. Turbulence, shocks, and bulk motions within the ICM can be caused by processes such as galaxy mergers,
active galactic nuclei (AGN) feedback, and the movement of galaxies through the medium.
These dynamics play a crucial role in the distribution of heat and metals within the ICM,
affecting the overall thermal state and evolution of the cluster.

The emission characteristics of the ICM are dominated by thermal bremsstrahlung (free-free emission), where high-energy electrons
are decelerated by ions, emitting X-rays in the process.
Additionally, line emission from highly ionized metals such as iron contributes to the X-ray spectrum.
Observing the ICM's X-ray emission provides critical information about the cluster's mass, temperature distribution,
and chemical composition, helping astronomers understand the physical processes governing galaxy clusters and their role in cosmic evolution.

Galaxies in Clusters
''''''''''''''''''''

Galaxies within galaxy clusters are diverse, ranging from massive elliptical galaxies to smaller spiral and irregular
galaxies. The distribution of these galaxies is not uniform; they tend to be more densely packed towards the cluster's
center and more sparsely distributed towards the edges. Elliptical galaxies, which are typically older and less active
in star formation, dominate the central regions and often form a distinct sequence in color-magnitude diagrams known as
the "red sequence." Spiral galaxies, which are more active in star formation, are often found in the outskirts of the
cluster where the gravitational influence is weaker.

The most massive and luminous galaxy in a cluster is the Brightest Cluster Galaxy (BCG),
typically located near the cluster's center. BCGs are often giant elliptical galaxies that have grown through
mergers and accretion of smaller galaxies. Their central position and significant mass make them key players in
the dynamics and evolution of the cluster. The BCG's immense gravitational pull can influence the movement and distribution
of other galaxies within the cluster, further shaping the cluster's overall structure. The prominence of the red sequence
highlights the evolutionary processes that favor the survival of older, redder galaxies in the cluster's dense environment.

.. note::

    For modeling purposes, it is almost never necessary to produce complex models of the galaxies within clusters. It is
    typical even to ignore the presence of stellar mass in these systems since stars make up :math:`\approx 1 \%` of the total
    mass budget. Nonetheless, there are some contexts in which the galaxies in the cluster can play an important role in the dynamics,
    and their inclusion should be carefully considered. This is particularly true for hydrodynamical simulations as the inclusion / exclusion of
    constituent galaxies will also impact stellar feedback processes and AGN feedback, which (in turn) have non-trivial implications for the
    resulting ICM physics.

Common Assumptions
''''''''''''''''''

In the context of modeling galaxy clusters, two assumptions are ubiquitous in the literature and extend in application to
the use of clusters as probes for fundamental physics.

- **Sphericity**: Because galaxy clusters are often dynamically relaxed and are formed stochastically (and thus generally don't
  have large rotational velocities) it is standard practice across the literature to assume galaxy clusters are spherical. From a theoretical
  perspective, this also affords considerable convenience.

  .. note::

    Observation of galaxy clusters illustrates (with seemingly increasing clarity) that the sphericity of galaxy clusters
    is not as good an assumption as one might hope. In many ways, the spherical treatment of these systems is an issue of convenience,
    not accuracy.

- **Dynamical Relaxation**: With the exception of recently merged clusters (or those actively undergoing mergers), it is common to
  assume galaxy clusters are dynamically relaxed. This has two major implications:

  - The collisionless components of the cluster (stars and dark matter) are virialized.
  - The ICM is in **hydrostatic equilibrium**; i.e. is fully pressure supported.

  As with the assumption of sphericity, more recent exploration indicates that many clusters (even those which were seemingly relaxed) are
  often still quite active.


Scaling Relations
'''''''''''''''''

- **M-T Relation**: :math:`M \propto T^{3/2}`

  This relation indicates that the total mass of a galaxy cluster scales with the temperature of the intracluster medium (ICM).
  Studies, such as those by Voit (2005) and Arnaud et al. (2005), have provided observational support for this relation, showing how
  higher temperature clusters are more massive due to their deeper gravitational potential wells.

- **L-T Relation**: :math:`L \propto T^{3}`

  The X-ray luminosity (L) of the cluster scales with the ICM temperature (T). This relation reflects the fact that hotter
  clusters emit more X-ray radiation. This empirical relationship has been validated through numerous observations,
  demonstrating a consistent increase in luminosity with higher temperatures.

- **Entropy-Temperature (S-T) Relation**: :math:`S \propto T^{2/3}`

  The entropy of the ICM scales with its temperature. This relation is important for understanding the thermal history
  of the gas within clusters, indicating how entropy is affected by heating and cooling processes.


Models of Galaxy Clusters
-------------------------

The purpose of the cluster generator library is to produce scientifically viable models of galaxy clusters for a variety of
different purposes. Below, we've compiled much of the critical theory that informs the code.

Fundamental Physics
'''''''''''''''''''

Gravity
+++++++

.. admonition:: Upcoming Changes

    CG will eventually support a variety of different gravitational paradigms. Once this feature has been added,
    the gravitational theory necessary to understand those parts of the code will be described here.

    For now, only Newtonian gravitation is used and it is assumed that the reader is sufficiently familiar with it.

The ICM
+++++++

Assuming the intracluster medium of galaxy clusters can be modeled as an
ideal fluid, the momentum density :math:`\rho{\bf v}` of the
gas obeys the Euler momentum equation (here written in conservative form
and ignoring magnetic fields, viscosity, etc.):

.. math::

    \frac{\partial({\rho_g{\bf v}})}{\partial{t}} + \nabla \cdot (\rho_g{\bf v}\otimes{\bf v})
    = -\nabla{P} + \rho_g{\bf g}

where :math:`\rho_g` is the gas density, :math:`{\bf v}` is the gas velocity,
:math:`P` is the gas pressure, and :math:`{\bf g}` is the gravitational
acceleration. The assumption of hydrostatic equilibrium implies that
:math:`{\bf v} = 0` and that all time derivatives are zero, giving:

.. math::

    \nabla{P} = \rho_g{\bf g}


In the context of galaxy clusters, we may typically assume an ideal gas behavior for the ICM, and therefore,

.. math::

    \nabla \Phi = -\frac{\nabla P}{\rho_g} = \frac{-k_b T}{m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

As such, we recognize that from :math:`\rho_g` and :math:`T`, we can generally obtain the acceleration and therefore the dynamical mass
of the system. Because the dynamical mass need not be equivalent to the gasseous mass, it follows that finding the dynamical mass will also
yield the halo mass of the system.

It is worth noting throughout the remainder of this document that for :math:`r\to \infty`, if :math:`\rho \sim r^\alpha` and :math:`T\sim r^\beta`,
then

.. math::

    \nabla \Phi \sim r^{\beta-1}.

This point becomes important as we begin discussing the interaction between the gas dynamics above and the potential of the system.

..
    Gravity
    '''''''

    In the preceding section, we described the basic mathematics of hydrostatic equilibrium; however, to find the dynamical mass and other related quantities,
    an expression for :math:`\nabla \Phi` must be obtained independently. In each gravitational theory implemented in CG, the dynamical field may be determined in terms of
    :math:`\Gamma` and can then be used to construct the correct mass profiles. In the window below, we've included 3 archetypal examples:

    .. tab-set::

        .. tab-item:: Newtonian

            In Newtonian gravity, the dynamical mass equation derived from the Poisson equation is

            .. math::

                M_{\mathrm{dyn}}(<r) = \frac{r^2}{G}\Gamma = \frac{-r^2 k_b T}{G m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

        .. tab-item:: AQUAL

            In the AQUAL flavor of MOND, the dynamical mass is different from the Newtonian case by a factor of :math:`\mu(a)`.

            .. math::

                M_{\mathrm{dyn}} = \frac{r^2}{G}\mu\left(\frac{| \Gamma |}{a_0}\right)\Gamma.

        .. tab-item:: QUMOND

            In the QUMOND flavor of MOND, the equations cannot be solved independently and therefore numerical methods are required to solve the
            implicit equation

            .. math::

                \nu\left(\frac{|\nabla \Psi |}{a_0}\right)\nabla \Psi = \Gamma,\;\;\nabla \Psi = \frac{GM_{\mathrm{dyn}}}{r^2}

    Each of these theories has a different asymptotic behavior in terms of the temperature. Therefore, the required temperature behavior at large
    radii is constrained by the following table. This should be considered when choosing a model.

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


The ICM is primarily composed of ionized hydrogen and helium, with a smaller fraction of heavier elements, or "metals."
The primordial hydrogen abundance in the ICM is a key factor in determining its overall composition. The baryonic matter in the universe,
which includes the ICM, has a hydrogen-to-helium ratio of about 12:1 by number,
reflecting the conditions of nucleosynthesis in the early universe. This primordial composition is modified by the
processes of stellar evolution and supernova explosions within the cluster galaxies, which enrich the ICM with metals like iron, oxygen, and silicon.

Regardless of metal-producing processes, it is generally sufficient to assume that the entirety of the ICM is H or He and to
treat the abundance of each as that predicted by Big Bang Nucleosynthsis (BBN).

Dark Matter
+++++++++++

Dark matter is a critical component in the models of galaxy clusters, comprising approximately 85% of the total mass of clusters.
Its gravitational influence dominates the dynamics and structure of clusters, despite being invisible to electromagnetic observations.
The presence of dark matter is inferred from its gravitational effects on visible matter, such as galaxies and the intracluster medium (ICM).
Dark matter's distribution within clusters is typically modeled using density profiles such as the Navarro-Frenk-White (NFW) profile:

.. math::

    \rho(r) = \frac{\rho_0}{x(1+x)^2},\;\;\text{where}\;\; x = \frac{r}{r_s}.

Here, :math:`\rho_0` is a characteristic density and :math:`r_s` is a scale radius. The NFW profile describes a cuspy density
distribution that decreases with radius and fits well with observations and simulations of dark matter halos.

The gravitational potential generated by dark matter is essential for maintaining the hydrostatic equilibrium of the ICM.
It provides the necessary gravitational pull that balances the thermal pressure of the hot gas. Additionally, the distribution of dark matter
influences the overall shape and potential well of the cluster, affecting the orbits of galaxies and the dynamics of the ICM.
Accurate modeling of dark matter is crucial for deriving cluster masses from X-ray and gravitational lensing observations.
These models help to understand the large-scale structure of the universe and the role of dark matter in galaxy formation and evolution.

Spherical Models
''''''''''''''''

Despite the limitations of treating galaxy clusters as idealized, spherical systems, the practice of doing so remains ubiquitous for
observational and theoretical work involving galaxy clusters. As such, cluster generators core code revolves around producing such model systems
from any number of underlying physical assumptions.

When generating spherical models, the user generally needs to provide at least 2 physical profiles; the rest can be determined from
theory.

.. hint::

    The entire process of producing a model revolves around hydrostatic equilibrium. Generally, the user provides a thermodynamic property (typically gas density) and
    a gravitational property (total mass, gravitational field, etc.) from which all the other properties may be deduced.

Cluster Generator users will therefore need to begin by creating the necessary :py:class:`radial_profiles.RadialProfile` instances to represent
these basis profiles.


Generating a :py:class:`model.ClusterModel` Using Radial Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The user may generate a :py:class:`model.ClusterModel` from the :py:meth:`model.ClusterModel.from_dens_and_temp` method, which requires :py:class:`radial_profiles.RadialProfile` objects
for the temperature and the gas density. Using the condition of hydrostatic equilibrium, the potential, pressure, dynamical mass, and other
necessary fields are automatically computed based on the chosen gravity theory.

The pressure :math:`P(r)` is provided via the ideal gas law:

.. math::

    P(r) = \frac{\rho_g(r) k_b T(r)}{m_p \eta},

where :math:`\eta` is the mean-molecular mass (generally 0.6 for galaxy clusters).

Once the pressure is determined, Euler's Equations can be used for an incompressible fluid, yielding

.. math::

     \frac{-\nabla P(r)}{\rho_g} = \nabla \Phi

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the case where the dynamical density is already known, the halo mass function is trivially obtained from

.. math::

    \rho_{\mathrm{halo}} = \rho_{\mathrm{dyn}} - \rho_{\mathrm{bary}},

where the baryonic component may contain a stellar distribution if the user supplies one. The difficult in this case
is to efficiently determine the temperature profile which yields HSE. Recalling that

.. math::

    \nabla \Phi = -\frac{\nabla P}{\rho_g} = \frac{-k_b T}{m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

The differential equation may be inverted, yielding

.. math::

    T(r) = \frac{-m_p \eta}{k_b \rho_g} \int_{r_0}^r \rho_g(r') \nabla \Phi(r') dr' + \frac{\rho_g(r_0)}{\rho_g(r)}T(r_0).

The most efficient approach here is to take :math:`r_0 = \infty`, in which case,

.. math::

    T(r) = \frac{m_p \eta}{k_b \rho_g} \int_{r}^\infty \rho_g(r') \nabla \Phi(r') dr'.

From this, the temperature profile is obtained.

``ClusterModel`` from Gas Density and Gas Entropy Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Users may also generate a :py:class:`model.ClusterModel` object from an entropy profile instead of the corresponding temperature profile.
There is a 1-to-1 correspondence between entropy and temperature given that the gas density is fixed, therefore, the procedure
follows exactly from the previous section; however, the entropy is converted first to a temperature profile using the widely accepted formula

.. math::

    S(r) = k_b T_g(r)n_e(r)^{-2/3}.


``ClusterModel`` Without Gas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all of the :py:class:`model.ClusterModel` objects are generated with a gas density profile. If one wants to produce a gas-less model,
the :py:meth:`model.ClusterModel.no_gas` method is available. Users must provide a dynamical density profile, and are optionally allowed
to provide a stellar component. See the API reference for syntactical information.

Checking the Hydrostatic Equilibrium
+++++++++++++++++++++++++++++++++++++
In the vast majority of cases, the aim of the :py:class:`model.ClusterModel` object is to produce a model which is in hydrostatic equilibrium. To check how successful the algorithm was in
generating a hydrostatic galaxy cluster, one can use the :py:meth:`model.ClusterModel.check_hse` method, which will provide the maximal relative deviation of the system from hydrostatic equilibrium.

This is done by recalling that the hydrostatic equilibrium condition requires that

.. math::

    -\frac{\nabla P}{\rho_g} = \nabla \Phi.

This, we define the hydrostatic variable :math:`\xi` such that

.. math::

    \xi = \frac{\nabla \Phi \rho_g + \nabla \Phi}{\nabla \Phi \rho_g}.

clearly, :math:`\xi \approx 0` indicates a successfully equilibrated cluster.



Galactic Dynamics and Virialization
'''''''''''''''''''''''''''''''''''

Collisionless components of astrophysical systems are subject to their own stability constraints in much the same way
that the ICM is. In order to incorporate dark matter and stellar components into cluster generator's models, the velocities of the
corresponding particles must be selected so that the particular structure of the model is stable (as an ensemble) with time. This stability,
referred to as virial equilibrium, is one of the trickiest aspects of preparing initial conditions for hydrodynamical simulations. The theory is presented here:

The mass density :math:`\rho({\bf r})` of such a system can be derived by
integrating the phase-space distribution function :math:`f({\bf r}, {\bf v})`
over velocity space:

.. math::

    \rho({\bf r}) = \int{f({\bf r}, {\bf v})d^3{\bf v}}

where :math:`{\bf r}` and :math:`{\bf v}` are the position and velocity
vectors.

Spherically Symmetric Systems
+++++++++++++++++++++++++++++

Assuming spherical symmetry and isotropy, all quantities are simply
functions of the scalars :math:`r` and :math:`v`, and
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
determine particle speeds. For our cluster models, this equation must (in
general) be solved numerically, even if the underlying dark matter, stellar,
and gas densities can be expressed analytically.

To generate the particle speeds, the distribution function :math:`f({\cal E})`
is used with uniform random numbers :math:`u \in [0, 1]` via an
acceptance-rejection method. The particle velocity components are isotropically
distributed in the tangential directions :math:`\theta` and :math:`\phi`.

The Local Maxwellian Approximation
++++++++++++++++++++++++++++++++++

.. admonition:: Upcoming Feature

    For spherically symetric systems, the Eddington formula is the obvious approach; however, non-Newtonian gravitational
    systems and non-spherical systems do not follow it. As we add more features, we will describe more sophisticated approximation methods.

Checking the Virial Equilibrium
+++++++++++++++++++++++++++++++

It is probably a good idea to check that the resulting distribution functions
for the dark matter and/or stars are consistent with the input mass density
profiles. The :py:meth:`~cluster_model.ClusterModel.check_dm_virial`
or :py:meth:`~cluster_model.ClusterModel.check_star_virial`
methods can be used to perform a quick check on the accuracy of the virial
equilibrium model for each of these types. These methods return two NumPy
arrays, the first being the density profile computed from integrating the
distribution function, and the second being the relative difference between
the input density profile and the one calculated using this method.

.. code-block:: python

    import matplotlib.pyplot as plt
    rho, diff = p.check_dm_virial()
    # Plot this up
    fig, ax = plt.subplots(figsize=(10,10))
    ax.loglog(vir["radius"], vir["dark_matter_density"], 'x',
              label="Input mass density", markersize=10)
    ax.loglog(vir["radius"], rho, label="Derived mass density", lw=3)
    ax.legend()
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("$\mathrm{\\rho\ (M_\odot\ kpc^{-3})}$")

.. image:: _images/check_density.png

One can see that the derived density diverges from the input density at large
radii, due to difficulties with numerically integrating to infinite radius. So long
as the maximum radius of the profile is very large, this should not matter very
much.

Advanced Modeling
-----------------

.. admonition:: Upcoming Changes

    Future additions to this project, including non-spherical models, non-thermal pressure support, etc. will all be
    described in this section.

Magnetic Fields
---------------

Magnetic fields play a major role in the dynamics of the ionized plasma modeled in hydro simulations of galaxy clusters. In ``cluster_generator``, there
are currently two approaches for generating a magnetic field for a galaxy cluster.

Setting a Magnetic Field from :math:`\beta`
''''''''''''''''''''''''''''''''''''''''''''
Users can specify the magnetic field profile using the :py:meth:`~model.ClusterModel.set_magnetic_field_from_beta`, which allows the user
to specify the magnetic pressure factor :math:`\beta`, which is defined such that

.. math::

    \beta = \frac{p_{\mathrm{thermal}}}{p_{\mathrm{magnetic}}}.

Setting a Magnetic Field from plasma density
''''''''''''''''''''''''''''''''''''''''''''
Another common approach for initializing magnetic fields is to let the magnetic field be proportional to a power of the gas density. Most commonly, this value is
:math:`\eta = \frac{2}{3}`; however, the user may specify whichever exponent they choose. To initialize a cluster with a magnetic field using this approach, use the :py:meth:`~model.ClusterModel.set_magnetic_field_from_density`.
