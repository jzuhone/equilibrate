.. _gravity:

Gravity Theories
----------------

The CGP provides access not only to Newtonian gravity models for galaxy clusters, but also to a variety of modified
gravity options. On this page, we provide brief summaries of the different gravitational theories provided and also describe
how one goes about implementing a new gravitational mode into the CGP.

.. contents::

.. raw:: html

   <hr style="height:5px;background-color:black">

Available Gravity Theories
++++++++++++++++++++++++++

.. card-carousel:: 2

    .. card:: Newtonian Gravity
        :link: newtonian-gravity
        :link-type: ref

        **Type**: Classical
        ^^^
        Standard implementation of Newtonian gravity.
        +++
        | **Class**: :py:class:`gravity.NewtonianGravity`

    .. card:: AQUAL Gravity
        :link: aqual-gravity
        :link-type: ref

        **Type**: MONDian
        ^^^
        A classical MOND implementation with a non-linear aquadratic field lagrangian.
        +++
        | **Source**: `1984ApJ...286....7B <https://ui.adsabs.harvard.edu/link_gateway/1984ApJ...286....7B/ADS_PDF>`_
        | **Class**: :py:class:`gravity.AQUALGravity`
    .. card:: QUMOND Gravity
        :link: qumond-gravity
        :link-type: ref

        **Type**: MONDian
        ^^^
        A classical MOND implementation with 2 scalar fields and two linear (modified) poisson equations.
        +++
        | **Source**: `2010MNRAS.403..886M <https://ui.adsabs.harvard.edu/link_gateway/2010MNRAS.403..886M/EPRINT_PDF>`_
        | **Class**: :py:class:`gravity.QUMONDGravity`
    .. card:: EMOND Gravity
        :link: emond-gravity
        :link-type: ref

        **Type**: MONDian
        ^^^
        A semi-classical extension of MOND which allows the interpolation constant to be a function of potential.
        +++
        | **Source**: `2010MNRAS.403..886M <https://ui.adsabs.harvard.edu/link_gateway/2010MNRAS.403..886M/EPRINT_PDF>`_
        | **Class**: :py:class:`gravity.EMONDGravity`
.. raw:: html

   <hr style="height:2px;background-color:black">

Using Non-Newtonian Gravity
+++++++++++++++++++++++++++

Using non-newtonian gravity theories is simple! When you're creating a new :py:class:`model.ClusterModel` object, you can simply
specify the kwargs ``gravity="<Gravity Name>"`` to use a different gravity in the construction process. The name should be the exact same
as the class name with ``Gravity`` removed. For example, to use :py:class:`gravity.AQUALGravity`, use ``gravity="AQUAL"`` in your code.

Writing Your Own Gravity
++++++++++++++++++++++++

Copying the Template
````````````````````
To create your own gravity theory, the first step is to copy the gravity template present in the :py:mod:`gravity` module. Inside of the
``gravity.py`` file, there is a ``_Template`` class which should be copied to provide some of the core boiler-plate code for
your new gravity implementation. We have additionally included the template here:

.. code-block:: python

    class _Template(Gravity):
        # Configuring the classname #
        _classname = ""
        #: The available built-in potential solving methods.
        potential_methods = {
            1: ('_calcpo_gf', ["radius", "gravitational_field"]),
            2: ('_calcpo_drm', ["total_density", "radius", "total_mass"])
        }

        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)

        def potential(self, force=False):
            """
            Computes the gravitational potential of the :py:class:`model.ClusterModel` object that is connected to this instance.

            .. attention::

                This method passes directly to the class method :py:meth:`~gravity.NewtonianGravity.compute_potential`. If you only
                have fields and not a model, that is the better approach.

            Parameters
            ----------
            force: bool
                If ``True``, the potential will be recomputed even if it already exists.

            Returns
            -------
            None

            See Also
            --------
            :py:meth:`gravity.NewtonianGravity.compute_mass`
            :py:meth:`gravity.NewtonianGravity.compute_potential`

            """

            mylog.info(f"Computing gravitational potential of {self.model.__repr__()}.")

            if not force and self.is_calculated:
                mylog.warning(
                    "There is already a calculated potential for this model. To force recomputation, use force=True.")
                return None
            else:
                pass

            # - Pulling arrays
            self.model.fields["gravitational_potential"] = self.compute_potential(self.model.fields, spinner=True)

        @classmethod
        def compute_mass(cls, fields, attrs=None):
            pass

        @classmethod
        def compute_potential(cls, fields, attrs=None, spinner=True, method=None):
            r"""
            Computes the gravitational potential of the system directly from the provided ``fields``.

            .. attention::

                It is almost always ill-advised to call :py:meth:`gravity.NewtonianGravity.compute_potential` directly if you
                have an active instance of the object. The :py:meth:`gravity.NewtonianGravity.potential` method will compute
                the potential directly from the ``self.model`` object if an instances exists. This method should be reserved for cases when it
                is undesirable to have to construct a full system.

            Parameters
            ----------
            fields: dict
                The fields from which to compute the potential.
            attrs: dict
                Additional attributes to pass. These would match the instance's attributes if this were a fully realized
                instance of the class.
            spinner: bool
                ``False`` to disable the spinners.

            Returns
            -------
            unyt_array
                The computed gravitational potential of the system.

            """
            if attrs is None:
                attrs = {}

            attrs["spinner"] = spinner

            if method is None:
                method = cls._choose_potential_method(fields)

                if method is None:
                    raise ValueError(
                        "Failed to find a valid computation method for the potential using the provided fields.")

            eprint(f"Computing with method={method}", n=2)
            return getattr(cls, method)(fields, **attrs)

        @classmethod
        def _calcpo_gf(cls, fields, **kwargs):
            pass

        @classmethod
        def _calcpo_drm(cls, fields, **kwargs):
            pass

Once you have copied the necessary template, you should rename it with a proper name following the convention of other gravity
theories. Make sure to set the ``classname`` in the header of the class.

Defining Methods
````````````````

There are 3 important methods you need to include in your code for it to function: ``_calcpo_drm``, ``_calcpo_gf`` and ``compute_mass``. The first two should
compute the potential from density, radius, and mass or from the gravitational field respectively. Finally, the ``compute_mass`` method should compute the dynamical mass of
the system from the potential field.