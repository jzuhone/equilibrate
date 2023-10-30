.. _correction_methods:

Methods For Correcting Non-Physical Regions
===========================================

Type 0 NPRs
-----------

As a matter of convention, Type 0 NPRs are never corrected in CG because the represent
a fundamental failure in the user's initialization parameters and there is no clear physical foundation for even
providing a method to fix them.

Type 1 NPRs
-----------

.. tab-set::

    .. tab-item:: Type 1a NPRs

        .. note::

            Recall that :py:class:`correction.Type1aNPR` are caused by inconsistent slopes of the profiles.

        To correct these non-physical regions, a positive interpolation scheme is used. The gravitational field :math:`\nabla \Phi`
        is computed using the original profiles and the holes are found using the Hole-Finding Algorithm (See :py:mod:`numalgs`). These
        holes correspond to the regions in which the underlying profiles are inconsistent. To fix the holes, the :math:`\nabla \Phi` profile is
        patched with :math:`C^1[a,b]` splines in such a way as to force the profile to be strictly positive. The ``correction_parameter`` kwarg can be
        passed to the correction method to determine the degree of severity in the correction. If :math:`\gamma_c = 1`, then the corresponding interpolation
        scheme will default to the monotone interpolation scheme, if :math:`\gamma_c = 0`, then the profile will be allowed to dip significantly toward zero within
        the region of the hole but never go negative.

        .. figure:: _images/numerical_algorithms/test_correction_NRP1a.h5_corrected.png

            A naive generation of the cluster A133 using the work of [ViKrF06]_ to construct best fit profiles (red) and its corrected
            version (blue). The underlying cause of this NPR is the steep slope of the temperature profile, which leads to the hole in
            the gravitational field.

        .. warning::

            There are instances in which a particularly high correction parameter will make it intractable to actually
            find a viable domain of interpolation.

Type 2 NPRs
-----------

.. note::

    Pardon our dust, we're still working on this section of the documentation.


References
----------

.. [ViKrF06] Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691.
