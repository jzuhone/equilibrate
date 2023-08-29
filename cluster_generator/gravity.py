r"""
Gravity
=======
The :py:mod:`gravity` module contains the :py:class:`gravity.Gravity` class, which essentially acts as a wrapper for
solving the Poisson equation in each of the available gravitational theories. The various available gravitational theories are
implemented in classes inheriting from the base :py:class:`gravity.Gravity` class.
"""
import sys

import numpy as np
from halo import Halo
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from unyt import unyt_array, unyt_quantity

from cluster_generator.utils import \
    integrate, mylog, G, integrate_toinf, truncate_spline, log_string

# -------------------------------------------------------------------------------------------------------------------- #
# Constants ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
#: The MOND gravitational acceleration constant.
a_0 = unyt_quantity(1.2e-10, "m/s**2")


# -------------------------------------------------------------------------------------------------------------------- #
# Classes ============================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
class Gravity:
    """
    The ``Gravity`` class is the basis class for custom gravity implementations in ``cluster_generator``.
    
    Parameters
    ----------
    model: cluster_generator.model.ClusterModel
        The model on which this ``Gravity`` instance will be computing the potential.
    
    
    """
    _classname = "BaseGravity"

    #  DUNDER METHODS
    # ----------------------------------------------------------------------------------------------------------------- #
    def __init__(self, model):
        #: The model being computed by this instance.
        self.model = model
        self.attrs = {}

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
        mylog.info(f"Resetting {self}...")
        self.model["gravitational_field"] = None


class NewtonianGravity(Gravity):
    """
    The Newtonian Gravity instance of :py:class:`gravity.Gravity`
    """
    # Configuring the classname #
    _classname = "Newtonian"

    def __init__(self, model, **kwargs):
        # -- Providing the inherited structure -- #
        super().__init__(model)

        # -- Checking for necessary attributes -- #
        self.attrs = kwargs

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
        # - Pulling arrays 
        rr = self.model.fields["radius"].d

        mylog.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={self._classname}.")
        tdens_func = InterpolatedUnivariateSpline(rr, self.model.fields["total_density"].d)

        # - Computing - #
        gpot_profile = lambda r: tdens_func(r) * r

        gpot1 = self.model.fields["total_mass"] / self.model.fields["radius"]
        gpot2 = unyt_array(4. * np.pi * integrate(gpot_profile, rr), "Msun/kpc")

        # - Finishing computation - #

        self.model.fields["gravitational_potential"] = -G * (gpot1 + gpot2)
        self.model.fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

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
            _val = -fields["radius"] ** 2 * fields["gravitational_field"] / G
            h.succeed(log_string("Dyn. Mass comp: [DONE] "))
        return _val


class AQUALGravity(Gravity):
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
    _classname = "AQUAL"

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
            self.attrs["a_0"] = a_0

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def potential(self, force=False):
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
        mylog.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={self._classname}.")

        # - pulling arrays
        rr = self.model["radius"].d
        tmass = self.model["total_mass"].d
        a_0 = self.attrs["a_0"].to("kpc/Myr**2").d

        # - Building the gamma function - #
        gamma_func = InterpolatedUnivariateSpline(rr, G.d * tmass / (a_0 * (rr ** 2)), k=2)

        # -- Redefining with an adjusted spline approach to prevent asymptotes from forming ---#
        # =====================================================================================#
        r_bound = 1.1 * rr[-1]
        gamma_func__adjusted = lambda x: np.piecewise(x,
                                                      [x <= r_bound,
                                                       x > r_bound],
                                                      [gamma_func,
                                                       lambda l: gamma_func(r_bound) * (r_bound / l) ** 2])

        self.model.fields["gamma"] = unyt_array(gamma_func__adjusted(rr))

        # - generating guess solution - #
        mylog.debug(f"[[Potential]] Creating AQUAL guess solution for implicit equation...")

        Gamma_func = lambda x: (1 / 2) * (
                gamma_func__adjusted(x) + np.sqrt(
            gamma_func__adjusted(x) ** 2 + 4 * gamma_func__adjusted(x)))  # -> big gamma del Phi / a_0
        _guess = Gamma_func(rr)

        # - solving - #
        mylog.debug(f"[[Potential]] Optimizing implicit solution...")
        _fsolve_function = lambda t: t * self.attrs["interp_function"](t) - self.model["gamma"]

        _Gamma_solution = fsolve(_fsolve_function, x0=_guess)

        Gamma = InterpolatedUnivariateSpline(rr, _Gamma_solution, k=2)

        # ** Defining the adjusted Gamma solution to prevent issues with divergence of the spline. **
        #
        #
        adj_Gamma = truncate_spline(Gamma, 0.95 * rr[-1], 7)

        # - Performing the integration process - #
        with Halo(text=log_string("Integrating Gamma profile..."), spinner="dots", stream=sys.stderr,
                  animation="marquee") as h:
            gpot2 = a_0 * unyt_array(integrate_toinf(adj_Gamma, rr), "(kpc**2)/Myr**2")
            h.succeed(text=log_string("Integrated Gamma profile."))
        # - Finishing computation - #
        self.model.fields["gravitational_potential"] = -gpot2
        self.model.fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

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
                attrs["a_0"] = a_0

            #  Returning output
            # -------------------------------------------------------------------------------------------------------- #
            h.succeed(log_string("Dyn. Mass comp: [DONE] "))
        return (-fields["radius"] ** 2 * fields["gravitational_field"] / G) * attrs["interp_function"](
            np.abs(fields["gravitational_field"].to("kpc/Myr**2").d) / attrs["a_0"].to("kpc/Myr**2").d)

    @staticmethod
    def _interp(x):
        return x / (1 + x)


class QUMONDGravity(Gravity):
    """
    QUMOND implementation of :py:class:`gravity.Gravity`.
    """
    _classname = "QUMOND"

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
            self.attrs["a_0"] = a_0

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
        # - Pulling arrays
        rr = self.model["radius"].to("kpc").d
        rr_max = 2 * rr[-1]
        tmass = self.model["total_mass"].to("Msun").d
        a_0 = self.attrs["a_0"].to("kpc/Myr**2").d
        ## -- Preparing for Execution -- ##
        mylog.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={self._classname}.")
        _gamma_func = InterpolatedUnivariateSpline(rr, (G.d * tmass) / (a_0 * (rr ** 2)), k=2)
        gamma_func = lambda r: _gamma_func(r) * (1 / (1 + (r / rr_max) ** 4))
        self.model["gamma"] = unyt_array(gamma_func(rr))
        gpot_profile = lambda r: - gamma_func(r) * a_0 * self.attrs["interp_function"](gamma_func(r))
        # - Performing the integration process - #
        gpot2 = unyt_array(integrate(gpot_profile, rr, 2 * rr[-1]), "(kpc**2)/Myr**2")
        # - Finishing computation - #
        self.model.fields["gravitational_potential"] = gpot2
        self.model.fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

    @classmethod
    def compute_mass(cls, fields, attrs={}):
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
                attrs["a_0"] = a_0
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

    @staticmethod
    def _interp(x):
        return ((1 / 2) * (
            np.sqrt(1 + (4 / x) + 1)) ** (
                    1))


# -------------------------------------------------------------------------------------------------------------------- #
# Catalog ============================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
# -- gravity catalog -- #
available_gravities = {
    "Newtonian": NewtonianGravity,
    "AQUAL"    : AQUALGravity,
    "QUMOND"   : QUMONDGravity
}
