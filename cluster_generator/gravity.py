r"""
Gravity
=======
The :py:mod:`gravity` module contains the :py:class:`gravity.Gravity` class, which essentially acts as a wrapper for
solving the Poisson equation in each of the available gravitational theories. The various available gravitational theories are
implemented in classes inheriting from the base :py:class:`gravity.Gravity` class.

.. admonition:: Info

    For more information on the underlying implementations, see :ref:`The Gravity Guide <gravity>`.

"""
import sys

import numpy as np
from halo import Halo
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from unyt import unyt_array

from cluster_generator.utils import \
    integrate, mylog, G, log_string, cgparams, devLogger,truncate_spline

# -------------------------------------------------------------------------------------------------------------------- #
# Constants ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
#: The MOND gravitational acceleration constant.
a0 = cgparams["gravity"]["general"]["mond"]["a0"]


# -------------------------------------------------------------------------------------------------------------------- #
# Classes ============================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
class Gravity:
    r"""
    The :py:class:`~gravity.Gravity` is the base class for different gravity implementations in ``cluster_generator``.
    This should typically not be called directly. Instead, select one of the fully constructed gravity theories from this module.

    Parameters
    ----------
    model: :py:class:`model.ClusterModel`
        The model to attach the :py:class:`gravity.Gravity` instance to.

    **kwargs
        Additional parameters to pass to the instance. These are stored in the ``self.attrs`` attribute.

    See Also
    --------
    :py:class:`gravity.NewtonianGravity`
    :py:class:`gravity.AQUALGravity`
    :py:class:`gravity.QUMONDGravity`
    :py:class:`gravity.EMONDGravity`
    """
    _classname = "BaseGravity"

    #  DUNDER METHODS
    # ----------------------------------------------------------------------------------------------------------------- #
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

    #  Properties
    # ----------------------------------------------------------------------------------------------------------------- #
    @property
    def is_calculated(self):
        """Will return ``True`` if there is already a value for the gravitational potential in place."""
        if "gravitational_potential" in self.model.fields and self.model.fields["gravitational_potential"] is not None:
            return True
        else:
            False

    #  Methods
    # ----------------------------------------------------------------------------------------------------------------- #
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
    Newtonian gravity implementation.

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
        # -- Providing the inherited structure -- #
        super().__init__(model, **kwargs)

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def potential(self, force=False):
        """
        Computes the gravitational potential of the object.

        .. attention::

            This method passes directly to the class method :py:meth:`~cluster_model.gravity.NewtonianGravity.compute_potential`. If you only
            have fields and not a model, that is the better approach.

        Parameters
        ----------
        force: bool
            If ``True``, the potential will be recomputed even if it already exists.

        Returns
        -------
        None
        """
        #  Logging
        # ------------------------------------------------------------------------------------------------------------ #
        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        #  Computing
        # ------------------------------------------------------------------------------------------------------------ #
        # - Pulling arrays
        self.model.fields["gravitational_potential"] = self.compute_potential(self.model.fields, spinner=True)

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
        unyt_array
            Computation result

        """

        return -fields["radius"] ** 2 * fields["gravitational_field"] / G

    @classmethod
    def compute_potential(cls, fields, attrs=None, spinner=True):
        """
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
        #  Managing inputs
        # ------------------------------------------------------------------------------------------------------------ #
        if attrs is None:
            attrs = {}

        #  Computing the potential
        # ------------------------------------------------------------------------------------------------------------ #
        devLogger.debug(f"Computing gravitational potential from {fields.keys()}. gravity={cls._classname}.")
        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:
            # - Pulling arrays
            rr = fields["radius"].d

            devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")
            tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)

            # - Computing - #
            gpot_profile = lambda r: tdens_func(r) * r

            gpot1 = fields["total_mass"] / fields["radius"]

            _v, errs = integrate(gpot_profile, rr)
            gpot2 = unyt_array(4. * np.pi * _v, "Msun/kpc")

            # - Finishing computation - #
            potential = -G * (gpot1 + gpot2)
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(log_string("Computed potential."))

        #  Managing warnings
        # ------------------------------------------------------------------------------------------------------------ #
        if len(errs) != 0:
            mylog.warning(f"Detected {len(errs)} warnings from integration. Likely due to physicality issues.")

            for wi in errs:
                devLogger.warning(wi.message)

        # -- Done -- #
        return potential

# -------------------------------------------------------------------------------------------------------------------- #
# MONDian Gravity Theories =====-===================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
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

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
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
        #  Logging
        # ------------------------------------------------------------------------------------------------------------ #
        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        #  Computing
        # ------------------------------------------------------------------------------------------------------------ #
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
        #  Managing attributes
        # ------------------------------------------------------------------------------------------------------------ #
        with Halo(text=log_string(f"Dyn. Mass comp; {cls._classname}."), stream=sys.stderr) as h:
            if attrs is None:
                attrs = {}

            if "interp_function" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `interp_function`. Setting to default...")
                attrs["interp_function"] = cls._interp

            if "a_0" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `a_0`. Setting to default...")
                attrs["a_0"] = a0

            #  Returning output
            # -------------------------------------------------------------------------------------------------------- #
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
        #  Logging and array construction
        # ------------------------------------------------------------------------------------------------------------ #
        devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")

        #  Managing required attributes
        # ------------------------------------------------------------------------------------------------------------ #
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

        # Carrying out computations
        # ------------------------------------------------------------------------------------------------------------ #
        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:

            # -- Pulling necessary arrays -- #
            rr = fields["radius"].d
            tmass = fields["total_mass"].d

            # -- Computing the gamma function (mu(Gamma)Gamma = gamma) -- gamma = newt_acceleration. -- #
            gamma_func = InterpolatedUnivariateSpline(rr, G.d * tmass / (a_0 * (rr ** 2)))
            fields["gamma"] = unyt_array(gamma_func(rr))
            gamma = gamma_func(rr)

            # - generating guess solution - #
            devLogger.debug(f"[[Potential]] Creating AQUAL guess solution for implicit equation...")
            guess_function = lambda x: (1 / 2) * (gamma + np.sqrt(gamma ** 2 + 4 * np.sign(gamma) * gamma))
            _guess = guess_function(rr)

            # - solving - #
            mylog.debug(f"[[Potential]] Optimizing implicit solution...")
            _fsolve_function = lambda t: t * attrs["interp_function"](t) - gamma

            _test_guess = _fsolve_function(_guess)

            if np.amax(_test_guess) <= cgparams["numerical"]["implicit"]["check_tolerance"]:
                _Gamma_solution = _guess
            else:
                _Gamma_solution = fsolve(_fsolve_function, x0=_guess,xtol=cgparams["numerical"]["implicit"]["solve_tolerance"])

            Gamma = InterpolatedUnivariateSpline(rr, _Gamma_solution)
            Gamma = truncate_spline(Gamma,rr[-1],7)
            _v, errs = integrate(Gamma, rr, rmax=2*rr[-1])
            gpot2 = a_0 * unyt_array(_v, "(kpc**2)/Myr**2")

            # - Finishing computation - #
            potential = -gpot2
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(text=log_string("Computed potential..."))

        #  Managing warnings
        # ------------------------------------------------------------------------------------------------------------ #
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

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
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
        #  Logging
        # ------------------------------------------------------------------------------------------------------------ #
        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        #  Computing
        # ------------------------------------------------------------------------------------------------------------ #
        # ------------------------#                                                                                #
        # In the far field, the acceleration in QUMOND goes as 1/r, which is not integrable. As such, our scheme   #
        # is to set the gauge at 2*rmax and then integrate out to it. Because we only need the potential out to rr #
        # its fine to not have an entirely integrated curve.                                                       #
        # -------------------------------------------------------------------------------------------------------- #
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
        #  Logging
        # ------------------------------------------------------------------------------------------------------------ #
        devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")

        #  Managing Attributes
        # ------------------------------------------------------------------------------------------------------------ #
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

        #  Computing
        # ------------------------------------------------------------------------------------------------------------ #
        with Halo(text=log_string("Computing potential..."), stream=sys.stderr,
                  enabled=(cgparams["system"]["text"]["spinners"] and spinner)) as h:

            # -- grabbing arrays -- #
            rr = fields["radius"].to("kpc").d

            # -- Computing the newtonian potential first -- #
            newtonian_potential = NewtonianGravity.compute_potential(fields, attrs=None, spinner=False)

            # -- determining the correct form of the proper potential -- #
            n_accel = np.gradient(newtonian_potential.d, rr)
            dphi = attrs["interp_function"](np.abs(n_accel) / a_0) * n_accel

            dphi_func = InterpolatedUnivariateSpline(rr, dphi)
            dphi_func = truncate_spline(dphi_func,rr[-1],7)
            _v, errs = integrate(dphi_func, rr, 2*rr[-1])
            gpot2 = unyt_array(_v, "kpc**2/Myr**2")

            # - Finishing computation - #
            potential = -gpot2
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(text=log_string("Computed potential..."))

        #  Managing warnings
        # ------------------------------------------------------------------------------------------------------------ #
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

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
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
        #  Logging
        # ------------------------------------------------------------------------------------------------------------ #
        mylog.info(f"Computing gravitational potential of {self.model.__repr__()}. gravity={self._classname}.")

        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if not force and self.is_calculated:
            mylog.warning(
                "There is already a calculated potential for this model. To force recomputation, use force=True.")
            return None
        else:
            pass

        #  Computing
        # ------------------------------------------------------------------------------------------------------------ #
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
        #  Managing attributes
        # ------------------------------------------------------------------------------------------------------------ #
        with Halo(text=log_string(f"Dyn. Mass comp; {cls._classname}."), stream=sys.stderr) as h:
            if attrs is None:
                attrs = {}

            if "interp_function" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `interp_function`. Setting to default...")
                attrs["interp_function"] = cls._interp

            if "a_0" not in attrs:
                mylog.debug(f"Gravity {cls._classname} requires kwarg `a_0`. Setting to default...")
                attrs["a_0"] = cls._base_a0

            #  Returning output
            # -------------------------------------------------------------------------------------------------------- #
            h.succeed(log_string("Dyn. Mass comp: [DONE] "))

        return (-fields["radius"] ** 2 * fields["gravitational_field"] / G) * attrs["interp_function"](
            np.abs(fields["gravitational_field"].to("kpc/Myr**2").d) / attrs["a_0"](fields["gravitational_potential"]))

    @classmethod
    def compute_potential(cls, fields, attrs=None, spinner=True,rmax=None):
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
        #  Logging and array construction
        # ------------------------------------------------------------------------------------------------------------ #
        from scipy.integrate import solve_ivp
        from cluster_generator.utils import truncate_spline
        devLogger.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={cls._classname}.")

        #  Managing required attributes
        # ------------------------------------------------------------------------------------------------------------ #
        # - providing a_0 - #
        if attrs is None:
            attrs = {}

        if "a_0" not in attrs:
            attrs["a_0"] = cls._base_a0

        # - providing the interpolation function - #
        attrs["interp_function"] = cls._interp


        # Carrying out computations
        # ------------------------------------------------------------------------------------------------------------ #
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
            gamma_func = truncate_spline(gamma_func,rr[-1],7)

            gamma_f = lambda x,p: gamma_func(x)/attrs["a_0"](p)

            fields["newtonian_field"] = unyt_array(gamma_func(rr))
            gamma = gamma_func(rr)


            # -- setting up lambda functions -- #
            devLogger.debug(f"[[Potential]] Creating EMOND differential equation...")
            Gamma_function = lambda x,p: attrs["a_0"](p)*((1+np.sqrt(1+4*(np.sign(gamma_f(x,p))/gamma_f(x,p))**alpha))**(1/alpha))*(np.sign(gamma_f(x,p))*np.abs(gamma_f(x,p)))/(2**(1/alpha))

            devLogger.debug(f"[[Potential]] Solving EMOND differential equation...")
            sol = solve_ivp(Gamma_function,(rmax,rr[0]),y0=[0],t_eval=rr[::-1],method="DOP853")

            gpot2 = unyt_array(sol.y[0,::-1], "(kpc**2)/Myr**2")

            # - Finishing computation - #
            potential = gpot2
            potential.convert_to_units("kpc**2/Myr**2")
            h.succeed(text=log_string("Computed potential..."))

        # -- DONE -- #
        return potential

    @classmethod
    def _interp(cls, x):
        return cls._interpolation_function(x)

# -------------------------------------------------------------------------------------------------------------------- #
# Catalog ============================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
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
