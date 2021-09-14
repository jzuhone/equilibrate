.. _radial_profiles:

Radial Profiles
---------------

To set up a cluster model in spherical hydrostatic and/or virial equilibrium,
one needs models for various quantities as a function of radius. 
:class:`~cluster_generator.radial_profiles.RadialProfile` objects 

General Profiles
================

Constant Profile
++++++++++++++++

.. math::

    p(r) = K

Power-Law Profile
+++++++++++++++++

.. math::

    p(r) = A\left(\frac{r}{r_s}\right)^\alpha

Density and Mass Profiles
=========================

NFW Profile
+++++++++++

.. math::

    \rho_{\rm NFW}(r) = \frac{\rho_s}{r/r_s\left(1+r/r_s\right)^2}

.. math::
    
    M_{\rm NFW}(<r) = 4\pi{\rho_s}{r_s^2}\left[\ln\left(1+\frac{r}{r_s}\right)-\frac{r}{r+r_s}\right]

"super-NFW" Profile
+++++++++++++++++++

.. math::

    \rho_{\rm sNFW}(r) = \frac{3M}{16\pi{a^3}}\frac{1}{r/a\left(1+r/a\right)^{5/2}}

.. math::

    M_{\rm sNFW}(<r) = M\left[1-\frac{2+r/a}{2(1+r/a)^{3/2}}\right]

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
