.. _radial_profiles:

Radial Profiles
---------------

Radial profiles of key physical quantities (temperature, density, etc.) form the backbone of CG's galaxy cluster models. Before
generating a model, the user must first initialize a set of representative radial profiles to form the foundation for the cluster.

In the ``cluster_generator`` code, these radial profiles are represented by :py:class:`radial_profiles.RadialProfile` objects.
These objects are effectively wrappers around standard functions of radius which provide some additional functionality. Additionally, CG
provides a number of pre-built radial profiles which can be used wtih ease. The profiles available in ``cluster_generator`` will now
be described, with the mathematical formulae given as well as an example instantiation.

General Profiles
================

These profiles fall under a general class of useful profiles and can be used to model any physical quantity of interest.

Constant Profile
++++++++++++++++

The :py:func:`~radial_profiles.constant_profile` creates a
profile which is constant with radius:

.. math::

    p(r) = K

where :math:`K` is a constant value with any units.

Example:

.. code-block:: python

    import cluster_generator as cg
    K = 1000.0 # constant
    p = cg.constant_profile(K)

Power-Law Profile
+++++++++++++++++

The :py:func:`~radial_profiles.power_law_profile` creates
a power-law profile.

.. math::

    p(r) = A\left(\frac{r}{r_s}\right)^\alpha

where :math:`A` is a normalization constant with any units, :math:`r_s` is a
scale radius with units of kpc, and :math:`\alpha` is a power-law index.

.. code-block:: python

    import cluster_generator as cg
    A = 1.0e-3 # normalization parameter
    r_s = 100.0 # scale radius in kpc
    alpha = -3.0 # index parameter
    p = cg.power_law_profile(A, r_s, alpha)

.. raw:: html

   <hr style="height:3px;background-color:black">

Density and Mass Profiles
=========================

These profiles are density and mass profiles from the literature that are
often used to model gas, DM, or total density/mass distributions.

NFW Profile
+++++++++++

The Navarro-Frenk-White (NFW) profile [1]_ is often used to model DM halos and
total density/mass profiles.

Density:

.. math::

    \rho_{\rm NFW}(r) = \frac{\rho_s}{r/r_s\left(1+r/r_s\right)^2}

Mass:

.. math::

    M_{\rm NFW}(<r) = 4\pi{\rho_s}{r_s^2}\left[\ln\left(1+\frac{r}{r_s}\right)-\frac{r}{r+r_s}\right]

where :math:`\rho_s` is a scale density in units of :math:`{\rm M_\odot~kpc^{-3}}`,
and :math:`r_s` is a scale radius in units of kpc.

.. admonition:: Mathematical Note

    The function :math:`M_{\mathrm{NFW}}(<r)` is manifestly divergent as :math:`r\to \infty`. As such, it is typically
    necessary to truncate the NFW profile at some maximal radius. See TNFW profile for more information.

An NFW density profile function can be generated using
:py:func:`~radial_profiles.nfw_density_profile`, and the NFW mass
profile function can be generated using
:py:func:`~radial_profiles.nfw_mass_profile`:

.. code-block:: python

    import cluster_generator as cg
    rho_s = 1.0e7 # scale density in units of Msun/kpc**3
    r_s = 100.0 # scale radius in kpc
    dp = cg.nfw_density_profile(rho_s, r_s)
    mp = cg.nfw_mass_profile(rho_s, r_s)

If you want to determine the scale density using a given concentration parameter,
you can use the :py:func:`~radial_profiles.nfw_scale_density`
function to determine it:

.. code-block:: python

    import cluster_generator as cg
    conc = 4.0 # - Example value for the concentration parameter.
    z = 0.8
    delta = 200
    rho_s = cg.radial_profiles.nfw_scale_density(conc,z=z,delta=delta)



"super-NFW" Profile
+++++++++++++++++++

The "super-NFW" profile [2]_ is similar to the NFW profile, except that it falls off faster at large
radius and thus its mass profile is finite at infinity.

Density:

.. math::

    \rho_{\rm sNFW}(r) = \frac{3M}{16\pi{a^3}}\frac{1}{r/a\left(1+r/a\right)^{5/2}}

Mass:

.. math::

    M_{\rm sNFW}(<r) = M\left[1-\frac{2+r/a}{2(1+r/a)^{3/2}}\right]

where :math:`M` is the total mass of the profile in units of
:math:`{\rm M_\odot}`, and :math:`a` is a scale radius in units of kpc.

An sNFW density profile function can be generated using
:py:func:`~radial_profiles.snfw_density_profile`, and the sNFW
mass profile function can be generated using
:py:func:`~radial_profiles.snfw_mass_profile`:

.. code-block:: python

    import cluster_generator as cg
    M = 1.0e15 # total mass of the halo in Msun
    a = 100.0 # scale radius in kpc
    dp = cg.snfw_density_profile(M, a)
    mp = cg.snfw_mass_profile(M, a)

Truncated NFW Profile
+++++++++++++++++++++
The Truncated NFW Profile (TNFW) is designed to fall off :math:`\sim r^{-2}` at radii beyond the truncation radius :math:`r_t`.
This causes the total mass of the profile to become finite. Typically, :math:`r_t` is set at some radius beyond the virial radius of
the cluster to minimize the impact that introducing the truncation has on the physics within the system of interest.

.. math::

    \rho_{\rm tNFW}(r) = \frac{\rho_s}{r/r_s\left(1+r/r_s\right)^2}\frac{1}{1+\left(r/r_t\right)^2}

Hernquist Profile
+++++++++++++++++
The Hernquist Profile [3]_ is a standard profile choice when modeling stellar bulges and shperical galaxies. It is regularly
used in the context of galaxy clusters to model the brightest central galaxy (BCG). The profile contains a logarithmic power-law slope
determined by the parameter :math:`\alpha`.

.. math::

    \rho_H(r) = \frac{M}{2\pi{a^3}}\frac{1}{r/a\left(1+r/a\right)^3}

.. math::

    M_H(<r) = M\frac{r^2}{(r+a)^2}

Einasto Profile
+++++++++++++++
The Einasto Profile [4]_ is another typical profile used for modeling spherical galaxies, bulges, and BCGs. The profile contains a logarithmic power-law slope
determined by the parameter :math:`\alpha`.

.. math::

    \rho_E(r) = {\rho_0}\exp\left[-\left(\frac{r}{h}\right)^\alpha\right]

where

.. math::

    \rho_0 = \frac{M}{4{\pi}h^3n\Gamma(3n)}

.. math::

    h = \frac{r_s}{d_n(n)^n}

.. math::

    d_n(n) = 3n - \frac{1}{3} + \frac{8}{1215n} + \frac{184}{229635n^2}

Vikhlinin et al. 2006 Density Profile
+++++++++++++++++++++++++++++++++++++
The Vikhlinin Density Profile [5]_ is a modified version of the standard :math:`\beta`-model [6]_, which aims to
replicate observed properties of clusters in the X-ray band. Modifications were made to create a cuspy core instead of a
flat core, parameterized by the :math:`\alpha` value. The second factor in the first term is added to increase the power-law slope at
large radii. Finally, the second term represents another :math:`\beta` model which increases the freedom of the model
near cluster cores.

.. math::

    \rho_{\rm V06}(r) = \rho_0\frac{(r/r_c)^{-\alpha/2}}{[1+(r/r_c)^2]^{3\beta/2-\alpha/4}}\frac{1}{[1+(r/r_s)^\gamma]^{\epsilon/2\gamma}}

Ascasibar & Markevitch 2006 Density Profile
+++++++++++++++++++++++++++++++++++++++++++
The AM06 Density Profile [7]_ may be derived as the hydrostatic equilibrium solution for a cluster having a temperature profile given
by the AM06 temperature profile.

.. math::

    \rho_{\rm AM06}(r) = \rho_0\left(1+\frac{r}{a_c}\right)\left(1+\frac{r}{ca_c}\right)^\alpha\left(1+\frac{r}{a}\right)^\beta

where

.. math::

    \alpha = -1-n\frac{c-1}{c-a/a_c}

.. math::

    \beta = 1-n\frac{1-a/a_c}{c-a/a_c}

.. raw:: html

   <hr style="height:3px;background-color:black">

Temperature Profiles
====================

Vikhlinin et al. 2006 Temperature Profile
+++++++++++++++++++++++++++++++++++++++++

.. math::

    T_{\rm V06}(r) = T_0t\frac{x+T_{\rm min}/T_0}{x+1}

where

.. math::

    x = \left(\frac{r}{r_{\rm cool}}\right)^{a_{\rm cool}}

.. math::

    t = \frac{(r/r_t)^{-a}}{[1+(r/r_t)^b]^{c/b}}

Ascasibar & Markevitch 2006 Temperature Profile
+++++++++++++++++++++++++++++++++++++++++++++++

.. math::

    T_{\rm AM06}(r) = \frac{T_0}{1+r/a}\frac{c+r/a_c}{1+r/a_c}

.. raw:: html

   <hr style="height:3px;background-color:black">
Entropy Profiles
================

Baseline Entropy Profile
++++++++++++++++++++++++

.. math::

    K(r) = K_0 + K_{200}\left(\frac{r}{r_{200}}\right)^\alpha

.. raw:: html

   <hr style="height:10px;background-color:black">
References
----------
.. [1] Navarro, J.F., Frenk, C.S.,& White, S.D.M. 1996, ApJ, 462, 563
.. [2] Lilley, E. J., Wyn Evans, N., & Sanders, J.L. 2018, MNRAS
.. [3] Astrophysical Journal v.356, p.359
.. [4] J. Einasto (1965), Kinematics and dynamics of stellar systems, Trudy Inst. Astrofiz. Alma-Ata 5, 87
.. [5] Vikhlinin, A., Kravtsov, A., Forman, W., Jones, C., Markevitch, M., Murray, S. S., & Van Speybroeck, L. (2006). Chandra sample of nearby relaxed galaxy clusters: Mass, gas fraction, and mass-temperature relation. The Astrophysical Journal, 640(2), 691.
.. [6] Cavaliere, A.&Fusco-Femiano, R.1978, A&A, 70, 677
.. [7] Ascasibar, Y., & Markevitch, M. (2006). The origin of cold fronts in the cores of relaxed galaxy clusters. The Astrophysical Journal, 650(1), 102.
