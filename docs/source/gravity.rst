.. _gravity:

Gravity Theories
----------------

The ``cluster_generator`` library allows users to build galaxy cluster models and initial conditions not only in Newtonian
gravity, but also in a couple of additional gravitational theories. On this page, summaries of all of the gravity theories included are provided.
Additional gravitational models can be easily written on the user side and a guide for doing so is also provided at the end.

Available Gravity Theories
++++++++++++++++++++++++++

.. card-carousel:: 2

    .. card:: Newtonian Gravity

        **Type**: Classical
        ^^^
        Standard implementation of Newtonian gravity.
        +++
        | **Class**: :py:class:`gravity.NewtonianGravity`

    .. card:: AQUAL Gravity
        :link: aqual
        :link-type: ref

        **Type**: MONDian
        ^^^
        A classical MOND implementation with a non-linear aquadratic field lagrangian.
        +++
        | **Source**: `1984ApJ...286....7B <https://ui.adsabs.harvard.edu/link_gateway/1984ApJ...286....7B/ADS_PDF>`_
        | **Class**: :py:class:`gravity.AQUALGravity`
    .. card:: QUMOND Gravity
        :link: qumond
        :link-type: ref

        **Type**: MONDian
        ^^^
        A classical MOND implementation with 2 scalar fields and two linear (modified) poisson equations.
        +++
        | **Source**: `2010MNRAS.403..886M <https://ui.adsabs.harvard.edu/link_gateway/2010MNRAS.403..886M/EPRINT_PDF>`_
        | **Class**: :py:class:`gravity.QUMONDGravity`
.. raw:: html

   <hr style="height:2px;background-color:black">


.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

.. _mond:
MONDian Gravity
+++++++++++++++
MOND (MOdified Newtonian Dynamics) is a set of gravitational theories which postulate a divergence from Newtonian gravity in
regions of low acceleration. First proposed by M. Milgrom [1]_ in 1983, MOND theories are universally held to 3 core-postulates:

1. At accelerations below some characteristic :math:`a_0`, MOND gravity diverges from the Newtonian theory.
2. In that asymptotic case,

.. math::

    \frac{a}{a_0}\textbf{a} = \textbf{g}_N

3. The transition between the Newtonian regime and the deep-MOND regime occurs over a range of acclerations around :math:`a_0`.

There are a variety of approaches by which to implement these postulates into a comprehensive classical field theory, several of which have been
implemented in ``cluster_generator``. Furthermore, many attempts are underway to establish self consistent covariant formulations of the
theory; however, there remains no standout candidate for a complete MOND theory.

.. _aqual:
AQUAL
=====
The AQUAL [3]_ (Aquadtratic Lagrangian) theory (``gravity = 'AQUAL'``) is typically considered the quintessential MOND theory, and has a Lagrangian of the form

.. math::

    \mathcal{L} = \frac{1}{2}\rho v^2 -\rho \Phi - \frac{a_0^2}{8\pi G} \mathcal{F}\left(\frac{|\nabla \Phi|^2}{a_0^2}\right),

which, upon variation over the field, yields a modified poisson equation

.. math::

    4\pi G \rho = \nabla \cdot \left[\mu\left(\frac{|\nabla \Phi|}{a_0}\right)\nabla\Phi\right].

Here :math:`\mu(x) = d\mathcal{F}(z)/dz`, where :math:`z = x^2`. To fulfill the postulates of the MOND paradigm, we require that
:math:`\mu(x) \to 1` as :math:`x \to \infty`, and :math:`\mu(x) \to x` as :math:`x\to 0`.

Implementation
^^^^^^^^^^^^^^

Cluster generator implements the AQUAL theory using the :py:class:`gravity.AQUALGravity` class, which contains the relevant poisson solver. As with all of the
spherically symmetric gravity theories implemented in this library, there are only two major alterations that have to be made: the potential and the dynamical mass.

| **Potential**:

In spherical symmetry, the Poisson equation for AQUAL simplifies to

.. math::

    \mu\left(\frac{|\nabla \Phi|}{a_0}\right)\nabla \Phi = \nabla \Psi,

where :math:`\Psi` is the Newtonian potential for the same system. Letting :math:`\gamma = \nabla \Psi /a_0` and :math:`\Gamma = \nabla \Phi / a_0`, this reduces to
:math:`\mu(|\Gamma|)\Gamma = \gamma`. As such, an implicit form of :math:`\Gamma` can be solved for and then integrated. In the :py:meth:`gravity.AQUALGravity.compute_potential` method,
we utilize a numerical solver to solve the implicit equation. To allow for optimal convergence, we first posit an approximate guess. To generate one, we assume :math:`\mu(x) = x/(1+x)`, which
is essentially the simplest interpolation function available. Then

.. math::

    \frac{\mathrm{sign}(\gamma) \Gamma^2}{1+ \mathrm{sign}(\gamma) \Gamma} = \gamma,

which leads to the guess

.. math::

    \Gamma_{\mathrm{guess}} = \frac{1}{2}\left\{\gamma + \sqrt{\gamma^2 + 4\mathrm{sign}(\gamma)\gamma}\right\}

We then integrate :math:`\Gamma` to obtain the complete solution.

.. attention::

    Because the deep-field behavior of MOND theories is to go as :math:`1/r`, the potential may not be treated as zero at infinity and then integrated inwards.
    Instead, we enforce a zero point of the potential at the edge of the system being initialized.

.. _qumond:

QUMOND
======
Unlike the AQUAL theory, QUMOND is a quasi-linear implementation of MOND [2]_. The Lagrangian takes the form

.. math::

    \mathcal{L} = \frac{1}{2}\rho v^2 - \rho \Phi - \frac{1}{8 \pi G}\left\{2\nabla \Phi \cdot \nabla \Psi -  a_0^2 \mathcal{Q}\left[\frac{|\nabla \Psi|^2}{a_0^2}\right]\right\}.

Variation over :math:`\Phi` yields the classical poisson equation:

.. math::

    4\pi G \rho_{\mathrm{dyn}} = \nabla^2\Psi.

As such, we identify the field :math:`\Psi` as the **Newtonian potential**. Variation over :math:`\Psi` yields a second Poisson equation of the form

.. math::

    4\pi G \hat{\rho} = \nabla^2\Phi = \nabla \cdot \left[\nu\left(\frac{|\nabla \Psi|}{a_0}\right)\nabla \Psi\right],

where :math:`\nu` is related to the function :math:`\mathcal{Q}` in much the same way that :math:`\mu` is related to :math:`\mathcal{F}` in the AQUAL theory.

In order to meet the necessary criteria for MOND's core postulates, :math:`\nu(x) \to 1, x\to \infty` and :math:`\nu(x) \to x^{-1/2}, x\to 0`.

.. admonition:: Mathematical Note

    In general, there is no reason why QUMOND and AQUAL would result in the same dynamics. To see this, consider that AQUAL's equations of motion reduce as

    .. math::

        \nabla \cdot \left\{\mu\left(\frac{a}{a_0}\right)\textbf{a} - \textbf{a}_N \right\} = 0, \implies \textbf{a} = \frac{1}{\mu(a/a_0)}\left[\textbf{a}_N + \nabla \times \textbf{H}_A\right],

    where :math:`\textbf{H}_A` is some vector field.

    Similarly, in QUMOND,

    .. math::

        \nabla \cdot \left\{\nu\left(\frac{a_N}{a_0}\right)\textbf{a}_N - \textbf{a}\right\} = 0, \implies \textbf{a} = \nu\left(\frac{a_N}{a_0}\right)\textbf{a}_N + \nabla \times \textbf{H}_Q.

    In general, the **curl-fields may be arbitrary** and need not be the same; however, in spherical symmetry, no vector field can have a curl. Thus

    .. math::

        \textbf{a}\mu\left(\frac{a}{a_0}\right) = \textbf{a}_N\;\;\text{AQUAL},\\ \nu\left(\frac{a_N}{a_0}\right)\textbf{a}_N = \textbf{a}\;\;\text{QUMOND}.

    What then is the relationship between :math:`\mu` and :math:`\nu` for which the observed dynamics are the same? Let :math:`x = a_N/a_0` and :math:`y=a/a_0`. Then

    .. math::

        \mu(x) = \nu(y)^{-1}.

    Given that :math:`y = x\mu(x)`, we recognize that

    .. math::

        \boxed{\mu(x)=\nu(x\mu(x))^{-1} \implies \mu(x)\nu(x\mu(x)) = 1.}

.. admonition:: Example

    The most commonly used interpolation functions are generally :math:`\mu(x) = x/(1+x^\alpha)^{1/\alpha}`. The corresponding :math:`\nu` can then be derived. In this case, we have to solve the equation

    .. math::

        \mu(y\nu(y)) = \frac{1}{\nu(y)}.\\\frac{y\nu(y)}{\left(1+(y\nu(y))^\alpha\right)^{1/\alpha}} = \frac{1}{\nu(y)}.\\
        \nu(y) = \left[\frac{1}{2}\left(1+\sqrt{1+\frac{4}{y^\alpha}}\right)\right]^{1/\alpha}.

    Throughout this reference, we call functions of this form **Milgromian Inverses**, and denote then as

    .. math::

        \mu(x) = \nu^\textbf{M}(x)

References
++++++++++
.. [1] Milgrom, M. 1983ApJ...270..365M
.. [2] Milgrom, M. 2010MNRAS.403..886M
.. [3] Bekenstein, J. Milgrom, M. 1984ApJ...286....7B