.. _radial_profiles:

Radial Profiles
---------------

To set up a cluster model in spherical hydrostatic and/or virial equilibrium,
one needs models for various quantities as a function of radius. 
:class:`~cluster_generator.radial_profiles.RadialProfile` objects 

The profiles available in ``cluster_generator`` will now be described, with
the mathematical formulae given as well as an example instantiation.

General Profiles
================

These profiles are for general use.

Constant Profile
++++++++++++++++

The :func:`~cluster_generator.radial_profiles.constant_profile` creates a
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

The :func:`~cluster_generator.radial_profiles.power_law_profile` creates
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

Density and Mass Profiles
=========================

These profiles are density and mass profiles from the literature that are
often used to model gas, DM, or total density/mass distributions.

NFW Profile
+++++++++++

The Navarro-Frenk-White (NFW) profile from Navarro, J.F., Frenk, C.S.,
& White, S.D.M. 1996, ApJ, 462, 563 is often used to model DM halos and
total density/mass profiles.

Density:

.. math::

    \rho_{\rm NFW}(r) = \frac{\rho_s}{r/r_s\left(1+r/r_s\right)^2}

Mass:

.. math::
    
    M_{\rm NFW}(<r) = 4\pi{\rho_s}{r_s^2}\left[\ln\left(1+\frac{r}{r_s}\right)-\frac{r}{r+r_s}\right]

where :math:`\rho_s` is a scale density in units of :math:`{\rm M_\odot~kpc^{-3}}`,
and :math:`r_s` is a scale radius in units of kpc. 

An NFW density profile function can be generated using 
:func:`~cluster_generator.radial_profiles.nfw_density_profile`, and the NFW mass
profile function can be generated using
:func:`~cluster_generator.radial_profiles.nfw_mass_profile`:

.. code-block:: python

    import cluster_generator as cg
    rho_s = 1.0e7 # scale density in units of Msun/kpc**3
    r_s = 100.0 # scale radius in kpc
    dp = cg.nfw_density_profile(rho_s, r_s)
    mp = cg.nfw_mass_profile(rho_s, r_s)
    
If you want to determine the scale density using a given concentration parameter, 
you can use the :func:`~cluster_generator.radial_profiles.nfw_scale_density`
function to determine it:



"super-NFW" Profile
+++++++++++++++++++

The "super-NFW" profile from Lilley, E. J., Wyn Evans, N., & Sanders, J.L. 2018,
MNRAS is similar to the NFW profile, except that it falls off faster at large
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
:func:`~cluster_generator.radial_profiles.snfw_density_profile`, and the sNFW 
mass profile function can be generated using
:func:`~cluster_generator.radial_profiles.snfw_mass_profile`:

.. code-block:: python

    import cluster_generator as cg
    M = 1.0e15 # total mass of the halo in Msun
    a = 100.0 # scale radius in kpc
    dp = cg.snfw_density_profile(M, a)
    mp = cg.snfw_mass_profile(M, a)

Truncated NFW Profile
+++++++++++++++++++++

.. math::

    \rho_{\rm tNFW}(r) = \frac{\rho_s}{r/r_s\left(1+r/r_s\right)^2}\frac{1}{1+\left(r/r_t\right)^2}

Hernquist Profile
+++++++++++++++++

.. math::

    \rho_H(r) = \frac{M}{2\pi{a^3}}\frac{1}{r/a\left(1+r/a\right)^3}

.. math::

    M_H(<r) = M\frac{r^2}{(r+a)^2}

Einasto Profile
+++++++++++++++

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

.. math::

    \rho_{\rm V06}(r) = \rho_0\frac{(r/r_c)^{-\alpha/2}}{[1+(r/r_c)^2]^{3\beta/2-\alpha/4}}\frac{1}{[1+(r/r_s)^\gamma]^{\epsilon/2\gamma}}

Ascasibar & Markevitch 2006 Density Profile
+++++++++++++++++++++++++++++++++++++++++++

.. math::

    \rho_{\rm AM06}(r) = \rho_0\left(1+\frac{r}{a_c}\right)\left(1+\frac{r}{ca_c}\right)^\alpha\left(1+\frac{r}{a}\right)^\beta

where 

.. math::

    \alpha = -1-n\frac{c-1}{c-a/a_c}

.. math::

    \beta = 1-n\frac{1-a/a_c}{c-a/a_c}

    
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

Entropy Profiles
================

Baseline Entropy Profile
++++++++++++++++++++++++

.. math::

    K(r) = K_0 + K_{200}\left(\frac{r}{r_{200}}\right)^\alpha
