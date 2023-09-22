r"""
Gravitational theories and their implementations in ``cluster_generator``.
"""
import sys

import numpy as np
from halo import Halo
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from unyt import unyt_array

from cluster_generator.utils import \
    integrate, mylog, G, log_string, cgparams, devLogger


#: The MOND gravitational acceleration constant.
a0 = cgparams["gravity"]["general"]["mond"]["a0"]



class Gravity:
    r"""
    The :py:class:`~gravity.Gravity` class is the base class for different gravity implementations in :py:mod:`cluster_generator`.
    This should typically not be called directly. Instead, select one of the fully constructed gravity theories from this module.

    Parameters
    ----------
    model: :py:class:`model.ClusterModel`
        The model to attach the :py:class:`gravity.Gravity` instance to.

    **kwargs
        Additional parameters to pass to the instance. These are stored in the :py:attr:`~gravity.Gravity.attrs` attribute.

    See Also
    --------
    :py:class:`gravity.NewtonianGravity`
    :py:class:`gravity.AQUALGravity`
    :py:class:`gravity.QUMONDGravity`
    :py:class:`gravity.EMONDGravity`
    """
    _classname = "BaseGravity"

    def __init__(self, model, **kwargs):
        #: The model attached to the gravity instance.
        self.model = model
        #: additional attributes attached to the instance
        self.attrs = kwargs

    def __repr__(self):
        return f"{self._classname} Instance"

    def __str__(self):
        return self.__repr__()

    def __contains__(self, item):
        return item in self.model.fields

    @property
    def is_calculated(self):
        """
        Determines if the associated :py:class:`model.ClusterModel` object has already had it's potential calculated.
        Returns
        -------
        bool
            ``True`` if the associated :py:class:`model.ClusterModel` object already has a ``"gravitational_potential"`` field populated. Otherwise, returns ``False``.
        """
        if "gravitational_potential" in self.model.fields and self.model.fields["gravitational_potential"] is not None:
            return True
        else:
            return False

    def reset(self):
        """
        Resets the :py:class:`model.ClusterModel` instance's `gravitation_potential` field to ``None``.

        Returns
        -------
        None
        """
        mylog.info(f"Resetting {self}...")
        self.model["gravitational_field"] = None


class NewtonianGravity(Gravity):
    """
    :py:class:`gravity.Gravity` implementation of Newtonian gravity. This class is the default gravity implementation.

    See Also
    --------
    :py:class:`gravity.Gravity`
    :py:class:`gravity.AQUALGravity`
    :py:class:`gravity.QUMONDGravity`
    :py:class:`gravity.EMONDGravity`

    """
    # Configuring the classname #
    _classname = "Newtonian"

    def __init__(self, model, **kwargs):
        """
        Initializes the :py:class:`gravity.NewtonianGravity` instance.

        Parameters
        ----------
        model: :py:class:`model.ClusterModel`
            The model to attach this instance to.

            .. note::

                If you do not have a :py:class:`model.ClusterModel` object to which this class should be attached, there is probably
                a static method or a class method which should serve your purpose without initialization.

        kwargs: dict, optional
            Additional keyword arguments to pass through the initialization process. These are transferred to the :py:attr:`gravity.Gravity.attrs` of the final object.
        """
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

        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

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
        r"""
        Computes the dynamical mass associated with the provided fields.


        Parameters
        ----------
        fields: dict of unyt_array
            Data fields to pass into the mass computation. Any keys may be included, but ``radius`` and ``gravitational_field`` are
            required keys.
        attrs: dict
            Additional attributes to use in computation.

        Returns
        -------
        unyt_array
            Computation result

        Notes
        -----
        This function is implemented by simply computing

        .. math::

            M_{\mathrm{dyn}}(<r) = -\frac{r^2 \nabla \Phi}{G}

        """
        del attrs
        return -fields["radius"] ** 2 * fields["gravitational_field"] / G

    @classmethod
    def compute_potential(cls, fields, attrs=None, spinner=True):
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

        Notes
        -----

        This computation exploits spherical symmetry by computing the gravitational potential in shells. Consider a
        spherical distribution of mass with density :math:`\rho(r)`. Consider the behavior at :math:`r = r'`. The gravitational potential from the mass
        within the radius :math:`r'` provides a potential

        .. math::

            \Phi_{in} = -\frac{4\pi G}{r}\int_0^{r} \xi^2 \rho(\xi) d\xi

        At each shell outward, the acceleration of that shell is

        .. math::

            \nabla \Phi_{\mathrm{shell}} = -\frac{GM(<r)}{r^2} = -4\pi G \rho(r) dr.

        Thus

        .. math::

            \Phi(r) = 4\pi G \left[\frac{1}{r}\int_0^r \xi^2 \rho(\xi) d\xi + \int_r^\infty \xi \rho(\xi) d\xi \right]
        """

        if attrs is None:
            attrs = {}

        del attrs

        devLogger.debug(f"Computing gravitational potential from {fields.keys()}. gravity={cls._classname}.")
        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:
            # - Pulling arrays
            rr = fields["radius"].d

            devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")
            tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)

            # - Computing - #
            gpot_profile = lambda xi: tdens_func(xi) * xi

            gpot1 = fields["total_mass"] / fields["radius"]

            _v, errs = integrate(gpot_profile, rr)
            gpot2 = unyt_array(4. * np.pi * _v, "Msun/kpc")

            # - Finishing computation - #
            potential = -G * (gpot1 + gpot2)
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(log_string("Computed potential."))

        if len(errs) != 0:
            mylog.warning(f"Detected {len(errs)} warnings from integration. Likely due to physicality issues.")

            for wi in errs:
                devLogger.warning(wi.message)

        # -- Done -- #
        return potential



class AQUALGravity(Gravity):
    """
    Implementation of the AQUAL (Aquadratic Lagrangian) formulation of MOND.

    See Also
    --------
    :py:class:`gravity.Gravity`
    :py:class:`gravity.NewtonianGravity`
    :py:class:`gravity.QUMONDGravity`
    :py:class:`gravity.EMONDGravity`
    """
    _classname = "AQUAL"
    _interpolation_function = cgparams["gravity"]["AQUAL"]["interpolation_function"]

    def __init__(self, model, **kwargs):
        # -- Providing the inherited structure -- #
        super().__init__(model, **kwargs)

        # -- forcing inclusion of the necessary attributes -- #
        if "interp_function" not in self.attrs:
            mylog.warning(f"Gravity {self._classname} requires kwarg `interp_function`. Setting to default...")
            self.attrs["interp_function"] = self._interp

        if "a_0" not in self.attrs:
            mylog.warning(f"Gravity {self._classname} requires kwarg `a_0`. Setting to default...")
            self.attrs["a_0"] = a0

    def potential(self, force=False):
        """
        Computes the gravitational potential of the object.

        .. attention::

            This method passes directly to the class method :py:meth:`~cluster_model.gravity.AQUALGravity.compute_potential`. If you only
            have fields and not a model, that is the better approach.

        Parameters
        ----------
        force: bool
            If ``True``, the potential will be recomputed even if it already exists.

        Returns
        -------
        None
        """

        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        self.model.fields["gravitational_potential"] = self.compute_potential(self.model.fields, attrs=self.attrs,
                                                                              spinner=True)

    @classmethod
    def compute_mass(cls, fields, attrs=None):
        """
        Computes the dynamical mass from the provided fields.

        Parameters
        ----------
        fields: dict
            The model fields associated with the object being computed.
        attrs: dict
            Additional attributes to use in computation.

        Returns
        -------
        np.ndarray
            Computation result

        """

        with Halo(text=log_string(f"Dyn. Mass comp; {cls._classname}."), stream=sys.stderr) as h:
            if attrs is None:
                attrs = {}

            if "interp_function" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `interp_function`. Setting to default...")
                attrs["interp_function"] = cls._interp

            if "a_0" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `a_0`. Setting to default...")
                attrs["a_0"] = a0

            h.succeed(log_string("Dyn. Mass comp: [DONE] "))

        return (-fields["radius"] ** 2 * fields["gravitational_field"] / G) * attrs["interp_function"](
            np.abs(fields["gravitational_field"].to("kpc/Myr**2").d) / attrs["a_0"].to("kpc/Myr**2").d)

    @classmethod
    def compute_potential(cls, fields, attrs=None, spinner=True):
        """
        Computes the gravitational potential of the system directly from the provided ``fields``.

        .. attention::

            It is almost always ill-advised to call :py:meth:`gravity.AQUALGravity.compute_potential` directly if you
            have an active instance of the object. The :py:meth:`gravity.AQUALGravity.potential` method will compute
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

        devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")

        # - providing a_0 - #
        if attrs is not None and "a_0" in attrs:
            a_0 = attrs["a_0"].to("kpc/Myr**2").d
        else:
            a_0 = a0.to("kpc/Myr**2").d

        # - providing the interpolation function - #
        if attrs is not None and "interp_function" in attrs:
            pass
        else:
            if attrs is None:
                attrs = {}
            attrs["interp_function"] = cls._interp

        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:

            # -- Pulling necessary arrays -- #
            rr = fields["radius"].d
            tmass = fields["total_mass"].d

            # -- Building computational arrays -- #
            x = np.geomspace(rr[0], 3 * rr[-1], 3 * len(rr))

            tmass_funct = InterpolatedUnivariateSpline(rr, tmass)
            t_mass_comp = tmass_funct(x)
            t_mass_comp[np.where(x > rr[-1])] = tmass[-1]

            # -- Computing the gamma function (mu(Gamma)Gamma = gamma) -- gamma = newt_acceleration. -- #
            gamma_func = InterpolatedUnivariateSpline(x, G.d * t_mass_comp / (a_0 * (x ** 2)))
            fields["gamma"] = unyt_array(gamma_func(rr))
            gamma = gamma_func(x)

            # - generating guess solution - #
            devLogger.debug(f"[[Potential]] Creating AQUAL guess solution for implicit equation...")
            guess_function = lambda y: (1 / 2) * (gamma + np.sqrt(gamma ** 2 + 4 * np.sign(gamma) * gamma))
            _guess = guess_function(x)

            # - solving - #
            mylog.debug(f"[[Potential]] Optimizing implicit solution...")
            _fsolve_function = lambda t: t * attrs["interp_function"](t) - gamma

            _test_guess = _fsolve_function(_guess)

            if np.amax(_test_guess) <= cgparams["numerical"]["implicit"]["check_tolerance"]:
                _Gamma_solution = _guess
            else:
                _Gamma_solution = fsolve(_fsolve_function, x0=_guess,
                                         xtol=cgparams["numerical"]["implicit"]["solve_tolerance"])

            Gamma = InterpolatedUnivariateSpline(x, _Gamma_solution)
            _v, errs = integrate(Gamma, rr, rmax=2 * rr[-1])
            gpot2 = a_0 * unyt_array(_v, "(kpc**2)/Myr**2")

            # - Finishing computation - #
            potential = -gpot2
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(text=log_string("Computed potential..."))

        if len(errs) != 0:
            mylog.warning(f"Detected {len(errs)} warnings from integration. Likely due to physicality issues.")

            for wi in errs:
                devLogger.warning(wi.message)

        # -- DONE -- #
        return potential

    @classmethod
    def _interp(cls, x):
        return cls._interpolation_function(x)


class QUMONDGravity(Gravity):
    """
    QUMOND implementation of :py:class:`gravity.Gravity`.

    See Also
    --------
    :py:class:`gravity.Gravity`
    :py:class:`gravity.NewtonianGravity`
    :py:class:`gravity.AQUALGravity`
    :py:class:`gravity.EMONDGravity`
    """
    _classname = "QUMOND"
    _interpolation_function = cgparams["gravity"]["QUMOND"]["interpolation_function"]

    def __init__(self, model, **kwargs):
        # -- Providing the inherited structure -- #
        super().__init__(model)

        # -- Checking for necessary attributes -- #
        self.attrs = kwargs

        if "interp_function" not in self.attrs:
            mylog.warning(f"Gravity {self._classname} requires kwarg `interp_function`. Setting to default...")
            self.attrs["interp_function"] = self._interp

        if "a_0" not in self.attrs:
            mylog.warning(f"Gravity {self._classname} requires kwarg `a_0`. Setting to default...")
            self.attrs["a_0"] = a0

    def potential(self, force=False):
        """
        Computes the potential array for the model to which this instance is attached.

        Parameters
        ----------
        force: bool
            If ``True``, a new computation will occur even if the model already has a potential field.

        Returns
        -------
        np.ndarray
            The relevant solution to the Poisson equation.

        """

        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        # ------------------------#                                                                                #
        # In the far field, the acceleration in QUMOND goes as 1/r, which is not integrable. As such, our scheme   #
        # is to set the gauge at 2*rmax and then integrate out to it. Because we only need the potential out to rr #

        self.model.fields["gravitational_potential"] = self.compute_potential(self.model.fields, attrs=self.attrs,
                                                                              spinner=True)

    @classmethod
    def compute_mass(cls, fields, attrs=None):
        """
        Computes the dynamical mass from the provided fields.

        Parameters
        ----------
        fields: dict
            The model fields associated with the object being computed.
        attrs: dict
            Additional attributes to use in computation.

        Returns
        -------
        np.ndarray
            Computation result

        """
        with Halo(text=log_string(f"Dyn. Mass comp; {cls._classname}."), stream=sys.stderr, enabled=False) as h:
            if attrs is None:
                attrs = {}

            if "interp_function" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `interp_function`. Setting to default...")
                attrs["interp_function"] = cls._interp

            if "a_0" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `a_0`. Setting to default...")
                attrs["a_0"] = a0
            # Compute the mass from QUMOND poisson equation

            # - Creating a function equivalent of the gravitational field - #
            _gravitational_field_spline = InterpolatedUnivariateSpline(fields["radius"].d,
                                                                       fields["gravitational_field"].to(
                                                                           "kpc/Myr**2").d /
                                                                       attrs["a_0"].to("kpc/Myr**2").d)

            _fsolve_function = lambda x: attrs["interp_function"](x) * x + _gravitational_field_spline(
                fields["radius"].d)

            # - Computing the guess - #
            _x_guess = np.sqrt(
                np.sign(fields["gravitational_field"].d) * _gravitational_field_spline(fields["radius"].d))

            # - solving - #
            _x = fsolve(_fsolve_function, _x_guess)
            h.succeed(log_string("Dyn. Mass comp: [DONE] "))
        return (attrs["a_0"] * fields["radius"] ** 2 / G) * _x

    @classmethod
    def compute_potential(cls, fields, attrs=None, spinner=True):
        """
        Computes the gravitational potential of the system directly from the provided ``fields``.

        .. attention::

            It is almost always ill-advised to call :py:meth:`gravity.QUMONDGravity.compute_potential` directly if you
            have an active instance of the object. The :py:meth:`gravity.QUMONDGravity.potential` method will compute
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
        from copy import deepcopy

        devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")

        # - providing a_0 - #
        if attrs is not None and "a_0" in attrs:
            a_0 = attrs["a_0"].to("kpc/Myr**2").d
        else:
            a_0 = a0.to("kpc/Myr**2").d

        # - providing the interpolation function - #
        if attrs is not None and "interp_function" in attrs:
            pass
        else:
            if attrs is None:
                attrs = {}
            attrs["interp_function"] = cls._interp

        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:

            rr = fields["radius"].to("kpc").d
            r_extension = np.geomspace(rr[-1], 3 * rr[-1], 2 * len(rr))[1:]
            rc = np.concatenate([rr, r_extension])

            cfields = deepcopy(fields)

            cfields["radius"] = unyt_array(rc, "kpc")
            normal = fields["total_density"].d[-1] * rr[-1] ** 3
            cfields["total_mass"] = unyt_array(np.concatenate([fields["total_mass"].d,
                                                               fields["total_mass"].d[-1] + (normal / 2) * (
                                                                           (1 / rr[-1] ** 2) - (
                                                                               1 / r_extension ** 2))]),
                                               fields["total_mass"].units)

            n_accel = G * cfields["total_mass"] / rc ** 2

            dphi = attrs["interp_function"](np.abs(n_accel.d) / a_0) * n_accel.d

            dphi_func = InterpolatedUnivariateSpline(rc, dphi)

            _v, errs = integrate(dphi_func, rr, rmax=2 * rr[-1])
            gpot2 = unyt_array(_v, "kpc**2/Myr**2")

            # - Finishing computation - #
            potential = -gpot2
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(text=log_string("Computed potential..."))

        if len(errs) != 0:
            mylog.warning(f"Detected {len(errs)} warnings from integration. Likely due to physicality issues.")

            for wi in errs:
                devLogger.warning(wi.message)

        # -- DONE -- #
        return potential

    @classmethod
    def _interp(cls, x):
        return cls._interpolation_function(x)


class EMONDGravity(Gravity):
    """
    Implementation of the Extended MOND formulation of MOND.

    See Also
    --------
    :py:class:`gravity.Gravity`
    :py:class:`gravity.NewtonianGravity`
    :py:class:`gravity.QUMONDGravity`
    :py:class:`gravity.AQUALGravity`
    """
    _classname = "EMOND"
    _interpolation_function = cgparams["gravity"]["AQUAL"]["interpolation_function"]
    _base_a0 = cgparams["gravity"]["EMOND"]["a0_function"]

    def __init__(self, model, **kwargs):
        # -- Providing the inherited structure -- #
        super().__init__(model, **kwargs)

        if "a_0" not in self.attrs:
            mylog.warning(f"Gravity {self._classname} requires kwarg `a_0`. Setting to default...")
            self.attrs["a_0"] = self._base_a0

    def potential(self, force=False):
        """
        Computes the gravitational potential of the object.

        .. attention::

            This method passes directly to the class method :py:meth:`~cluster_model.gravity.AQUALGravity.compute_potential`. If you only
            have fields and not a model, that is the better approach.

        Parameters
        ----------
        force: bool
            If ``True``, the potential will be recomputed even if it already exists.

        Returns
        -------
        None
        """

        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        self.model.fields["gravitational_potential"] = self.compute_potential(self.model.fields, attrs=self.attrs,
                                                                              spinner=True)

    @classmethod
    def compute_mass(cls, fields, attrs=None):
        """
        Computes the dynamical mass from the provided fields.

        Parameters
        ----------
        fields: dict
            The model fields associated with the object being computed.
        attrs: dict
            Additional attributes to use in computation.

        Returns
        -------
        np.ndarray
            Computation result

        """

        with Halo(text=log_string(f"Dyn. Mass comp; {cls._classname}."), stream=sys.stderr) as h:
            if attrs is None:
                attrs = {}

            if "interp_function" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `interp_function`. Setting to default...")
                attrs["interp_function"] = cls._interp

            if "a_0" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `a_0`. Setting to default...")
                attrs["a_0"] = cls._base_a0

            h.succeed(log_string("Dyn. Mass comp: [DONE] "))

        return (-fields["radius"] ** 2 * fields["gravitational_field"] / G) * attrs["interp_function"](
            np.abs(fields["gravitational_field"].to("kpc/Myr**2").d) / attrs["a_0"](fields["gravitational_potential"]))

    @classmethod
    def compute_potential(cls, fields, attrs=None, spinner=True, rmax=None):
        """
        Computes the gravitational potential of the system directly from the provided ``fields``.

        .. attention::

            It is almost always ill-advised to call :py:meth:`gravity.AQUALGravity.compute_potential` directly if you
            have an active instance of the object. The :py:meth:`gravity.AQUALGravity.potential` method will compute
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

        from scipy.integrate import solve_ivp
        from cluster_generator.utils import truncate_spline
        devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")

        # - providing a_0 - #
        if attrs is None:
            attrs = {}

        if "a_0" not in attrs:
            attrs["a_0"] = cls._base_a0

        # - providing the interpolation function - #
        attrs["interp_function"] = cls._interp

        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:

            # -- Pulling necessary arrays -- #
            rr = fields["radius"].d

            if not rmax:
                rmax = rr[-1]

            tmass = fields["total_mass"].d
            alpha = attrs["alpha"]

            # -- Computing the gamma function (mu(Gamma)Gamma = gamma) -- gamma = newt_acceleration. -- #
            gamma_func = InterpolatedUnivariateSpline(rr, G.d * tmass / (rr ** 2))
            gamma_func = truncate_spline(gamma_func, rr[-1], 7)

            gamma_f = lambda x, p: gamma_func(x) / attrs["a_0"](p)

            fields["newtonian_field"] = unyt_array(gamma_func(rr))
            gamma = gamma_func(rr)

            # -- setting up lambda functions -- #
            devLogger.debug(f"[[Potential]] Creating EMOND differential equation...")
            Gamma_function = lambda x, p: attrs["a_0"](p) * (
                        (1 + np.sqrt(1 + 4 * (np.sign(gamma_f(x, p)) / gamma_f(x, p)) ** alpha)) ** (1 / alpha)) * (
                                                      np.sign(gamma_f(x, p)) * np.abs(gamma_f(x, p))) / (
                                                      2 ** (1 / alpha))

            devLogger.debug(f"[[Potential]] Solving EMOND differential equation...")
            sol = solve_ivp(Gamma_function, (rmax, rr[0]), y0=[0], t_eval=rr[::-1], method="DOP853")

            gpot2 = unyt_array(sol.y[0, ::-1], "(kpc**2)/Myr**2")

            # - Finishing computation - #
            potential = gpot2
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(text=log_string("Computed potential..."))

        # -- DONE -- #
        return potential

    @classmethod
    def _interp(cls, x):
        return cls._interpolation_function(x)



# -- gravity catalog -- #
available_gravities = {
    "Newtonian": NewtonianGravity,
    "AQUAL"    : AQUALGravity,
    "QUMOND"   : QUMONDGravity
}

if __name__ == '__main__':
    from cluster_generator.radial_profiles import find_overdensity_radius, snfw_mass_profile, snfw_total_mass, \
        snfw_density_profile

    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    m = snfw_mass_profile(M, a)

    rmin, rmax = 0.1, 2 * r200
    r = np.geomspace(rmin, rmax, 1000)
    m, d, r = unyt_array(m(r), "Msun"), unyt_array(rhot(r), "Msun/kpc**3"), unyt_array(r, "kpc")

    QUMONDGravity.compute_potential({"total_mass": m, "total_density": d, "radius": r})
