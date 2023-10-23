"""
Methods for correcting non-physicality and other issues in CGP's :py:class:`model.ClusterModel` instances.


**Module Diagram**

.. inheritance-diagram:: cluster_generator.correction
    :parts: 1

"""
from collections.abc import Sized

import numpy as np
from tqdm.auto import tqdm

from cluster_generator.model import ClusterModel
from cluster_generator.numalgs import _check_non_positive
from cluster_generator.utils import LogMute, mylog


class NonPhysicalRegion:
    """
    Archetypal parent class for all :py:class:`correction.NonPhysicalRegion` instances. This class (and it's sub-classes)
    is used to denote the relevant non-physicalities in :py:class:`model.ClusterModel` instances.

    +--------------+---------+
    | Properties             |
    +==============+=========+
    | Scope        | All     |
    +--------------+---------+
    | Methods      | All     |
    +--------------+---------+
    | Gravity      | All     |
    +--------------+---------+
    | Correctable? | False   |
    +--------------+---------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1
    """

    _fields_of_interest = []
    _message = """
    Non-descript catch-all class of NPR. Progenitor class for all other NPR classes.
    """
    _scope = "generic"
    correctable = False

    def __init__(self, rmin, rmax, npr_type, obj):
        #: The minimum radius at which this defect is detected.
        self.rmin = rmin
        #: The maximum radius at which the defect is detected.
        self.rmax = rmax
        #: The type of non-physical region
        self.type = npr_type
        #: The progenitor object (for referencing)
        self.obj = obj

    def size(self):
        """Returns the size of the non-physical region."""
        return self.rmax - self.rmin

    def __repr__(self):
        return f"<NonPhysicalRegion (type={self.type}) [{np.format_float_scientific(self.rmin, 3)},{np.format_float_scientific(self.rmax, 3)}]>"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        print(self, other)
        if isinstance(other, (list, tuple)):
            return (self.type, self.rmin, self.rmax) == tuple(other)
        elif isinstance(other, self.__class__):
            return [self.type, self.rmin, self.rmax] == [
                other.type,
                other.rmin,
                other.rmax,
            ]
        else:
            raise TypeError(
                f"Cannot compare object of type {type(self)} and {type(other)}"
            )

    def is_close(self, other, rtol=1e-2):
        """
        Compares the two NPRs to check their relative separation is within tolerance.

        Parameters
        ----------
        other: NonPhysicalRegion, Iterable, dict
            If a :py:class:`correction.NonPhysicalRegion` instance is passed, the types are check to be equal and the values of the minimum
            and maximum radii are compared and required to be within ``rtol``. If a :py:class:`list` or :py:class:`tuple`, then it is assumed
            that the elements are of the form ``(type,rmin,rmax)`` and are compared as described above. If :py:class:`dict`, each of ``type``, ``rmin``, and ``rmax`` must
            be present as keys for comparison.
        rtol: float, optional
            The relative tolerance to require.

        Returns
        -------
        bool
        """
        if isinstance(other, self.__class__):
            if self.type != other.type:
                return False
            else:
                return (np.abs(self.rmin - other.rmin) / self.rmin < rtol) & (
                    np.abs(self.rmax - other.rmax) / self.rmax < rtol
                )
        elif isinstance(other, Sized):
            if len(other) != 3:
                return False
            else:
                if self.type != other[0]:
                    return False
                else:
                    return (np.abs(self.rmin - other[1]) / self.rmin < rtol) & (
                        np.abs(self.rmax - other[2]) / self.rmax < rtol
                    )

        elif isinstance(other, dict):
            if any(k not in other for k in ["type", "rmin", "rmax"]):
                return False
            else:
                return (
                    (np.abs(self.rmin - other["rmin"]) / self.rmin < rtol)
                    & (np.abs(self.rmax - other["rmax"]) / self.rmax < rtol)
                    & (self.type == other["type"])
                )
        else:
            raise TypeError(
                f"Cannot compare objects for closeness of type {type(self)} and {type(other)}"
            )

    def message(self, func=print):
        """
        Passes the NPR's message to output.
        Parameters
        ----------
        func: callable, optional
            The printing function to use to display the output. Default is ``print``.

        Returns
        -------
        None
        """
        func(
            f"""# ---- {self} ---- #
{self._message}
#-----------------------------------------------------------------------------#"""
        )

    @classmethod
    def identify(cls, model, recursive=True):
        """
        Identify the non-physicality of this type and return them.

        Parameters
        ----------
        model: :py:class:`model.ClusterModel`
            The :py:class:`model.ClusterModel` instances to identify NPRs in.
        recursive: bool
            If ``True``, seeks NPRs not just of this type, but of all subtypes of this type. This is propagated downward in
            a tree pattern.

        Returns
        -------
        list
            A list of instances of NPR objects indicating non-physical regimes of the cluster.
        """
        mylog.info(f"Seeking NPRs [scope={cls._scope}]...")
        nprs = []  # The NPRs being returned
        # -- Self-identification -- #
        nprs += cls._identify(model)
        # -- Subclass identification -- #
        if recursive and len(cls.__subclasses__()):
            mylog.info(
                f"Seeking NPRs in {len(cls.__subclasses__())} subclasses of {cls}"
            )
            for c in cls.__subclasses__():
                if hasattr(c, "identify"):
                    nprs += c.identify(model, recursive=recursive)

        return nprs

    @classmethod
    def correct(cls, model, recursive=True, fiter=10, **kwargs):
        r"""
        Identify and correct the non-physicalities in the provided :py:class:`model.ClusterModel` instance.

        Parameters
        ----------
        model: :py:class:`model.ClusterModel`
            The model to seek and correct non-physicalities in.
        recursive: bool, optional
            If ``True``, this function will iterate through all of the sub-classes of this NPR type and identify and correct the
            corresponding NPRs.
        fiter: int, optional
            The maximal number of times a correction will be attempted before the process returns in failure. Default = 10. The model
            may have more than ``fiter`` non-physicalities, but only ``fiter`` attempts will be made for any select non-physical region.
        **kwargs
            Additional kwargs to pass through to the correction methods in the sub-classes.

            +---------------------------+---------------------------------------------+-----------------------------------------+
            | Key                       | Information                                 | NPR types                               |
            +===========================+=============================================+=========================================+
            | ``correction_parameter``  | The degree of correction to impose during   | :py:class:`correction.Type1aNPR`        |
            |                           | interpolation procedures                    |                                         |
            +---------------------------+---------------------------------------------+-----------------------------------------+

        Returns
        -------
        :py:class:`model.ClusterModel`
            The corrected :py:class:`model.ClusterModel` instance.

        Raises
        ------
        CorrectionFailure
            If one of the non-physicalities is of an uncorrectable type.
        """
        mylog.info(f"Correcting non-physical behaviours in {model}.")
        with LogMute(mylog):
            nprs = cls.identify(model, recursive=recursive)

        prog_bar = tqdm(
            leave=True, total=len(nprs), desc=f"Correcting {len(nprs)} NPRs..."
        )
        n_iterations = 0

        with LogMute(mylog):
            while (len(nprs) > 0) and (n_iterations < fiter):
                lnprs = len(
                    nprs
                )  # keep track of the number of nprs then compare to see if progressing.
                try:
                    model = nprs[0]._correct(**kwargs)
                except CorrectionFailure as excep:
                    # gets raised because of non-fixable issues.
                    raise CorrectionFailure(
                        f"Failed to correct NPRs: {excep.__repr__()}"
                    )

                nprs = cls.identify(model, recursive=recursive)

                if lnprs > len(nprs):
                    prog_bar.update()
                    n_iterations = 0
                else:
                    n_iterations += 1

        if n_iterations >= fiter:
            raise CorrectionFailure(
                f"Failed to correct {nprs[0]} within {fiter} iterations."
            )
        else:
            mylog.info(f"Corrected NPRs in {model}.")
            return model

    def _correct(self):
        raise CorrectionFailure(
            f"There is no implemented resolution methodology for Type-{self._scope} NonPhysicalRegion objects: {self.__str__()}"
        )

    @classmethod
    def _identify(cls, model):
        return []


class Type0NPR(NonPhysicalRegion):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.NonPhysicalRegion`) which indicates NPRs of type 0.

    .. note::
        NPRs of type 0 are characterized by non-physicality inherent in the basic profiles of initialization. For example,
        a user trying to initialize with a negative temperature profile.

    +--------------+---------+
    | Properties             |
    +==============+=========+
    | Scope        | Type 0  |
    +--------------+---------+
    | Methods      | All     |
    +--------------+---------+
    | Gravity      | All     |
    +--------------+---------+
    | Correctable? | False   |
    +--------------+---------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1
    """

    _methods = []
    _fields_of_interest = []
    _message = """
    NPR of type 0: indicates that the user has erroneously specified a profile before any computation occurred.
    """
    _scope = "0"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, self._scope, obj)

    @classmethod
    def _identify(cls, model):
        nprs = []  # The NPRs being returned

        if "method" not in model.properties["meth"]:
            mylog.warning(
                "Failed to identify a 'generation_type' for the model. Unable to check for NPRs."
            )
        elif model.properties["meth"]["method"] not in cls._methods:
            pass
        else:
            for field in cls._fields_of_interest:
                if field in model.properties["meth"]["profiles"]:
                    domains = _check_non_positive(
                        model.properties["meth"]["profiles"][field](model["radius"].d),
                        domain=model.fields["radius"].d,
                    )
                if len(domains) != 0:
                    for d in domains:
                        nprs.append(cls(d[0], d[1], model))
                else:
                    pass

        if len(nprs) != 0:
            mylog.warning(
                f"Located {len(nprs)} non-physicalities of type {cls._scope}."
            )
        return nprs


class Type0aNPR(Type0NPR):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.Type0NPR`) which indicates NPRs of type 0a.

    .. note::
        NPRs of type 0a are characterized by non-physicality inherent in the basic profiles of initialization. In this case, negative temperature profiles.

    +--------------+-----------------------------+
    | Properties                                 |
    +==============+=============================+
    | Scope        | Type 0a                     |
    +--------------+-----------------------------+
    | Methods      | :math:`T_g + \rho_g`        |
    +--------------+-----------------------------+
    | Gravity      | All                         |
    +--------------+-----------------------------+
    | Correctable? | False                       |
    +--------------+-----------------------------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1
    """

    _methods = ["from_dens_and_temp"]
    _fields_of_interest = ["temperature"]
    _message = """
    NPR of type 0a: indicates that the user has erroneously specified a profile before any computation occurred. Specific to temperature profile.
    """
    _scope = "0a"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, obj)


class Type0bNPR(Type0NPR):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.Type0NPR`) which indicates NPRs of type 0b.

    .. note::
        NPRs of type 0b are characterized by non-physicality inherent in the basic profiles of initialization. In this case, negative density (total or gaseous) profiles.

    +--------------+-----------------------------+
    | Properties                                 |
    +==============+=============================+
    | Scope        | Type 0b                     |
    +--------------+-----------------------------+
    | Methods      | All                         |
    +--------------+-----------------------------+
    | Gravity      | All                         |
    +--------------+-----------------------------+
    | Correctable? | False                       |
    +--------------+-----------------------------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1
    """

    _fields_of_interest = ["density", "total_density"]
    _methods = ["from_dens_and_tden", "from_dens_and_temp"]
    _message = """
    NPR of type 0b: indicates that the user has erroneously specified a profile before any computation occurred. Specific to density profiles.
    """
    _scope = "0b"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, obj)


class Type0cNPR(Type0NPR):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.Type0NPR`) which indicates NPRs of type 0c.

    .. note::
        NPRs of type 0c are characterized by non-physicality inherent in the basic profiles of initialization. In this case, any initialization profile not
        considered in :py:class:`correction.Type0a` or :py:class:`correction.Type0b`.


    +--------------+-----------------------------+
    | Properties                                 |
    +==============+=============================+
    | Scope        | Type 0c                     |
    +--------------+-----------------------------+
    | Methods      | All                         |
    +--------------+-----------------------------+
    | Gravity      | All                         |
    +--------------+-----------------------------+
    | Correctable? | False                       |
    +--------------+-----------------------------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1
    """

    _methods = ["no_gas", "from_dens_and_entr", "from_arrays"]
    _fields_of_interest = ["density", "total_density", "temperature", "entropy"]
    _message = """
    NPR of type 0: indicates that the user has erroneously specified a profile before any computation occurred. Not elsewhere specified.
    """
    _scope = "0c"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, obj)


class Type1NPR(NonPhysicalRegion):
    r"""
    Non-Physical Region marker class (Subclass of :py:class:`correction.NonPhysicalRegion`) which indicates NPRs of type 1.

    .. note::
        NPRs of type 1 are gravitational theory independent regions of non-physicality driven by breakdowns in the derived
        behavior of the constructed profiles during initialization.


    +--------------+-----------------------------+
    | Properties                                 |
    +==============+=============================+
    | Scope        | Type 1                      |
    +--------------+-----------------------------+
    | Methods      | :math:`\rho_g + T_g`        |
    +--------------+-----------------------------+
    | Gravity      | All                         |
    +--------------+-----------------------------+
    | Correctable? | False                       |
    +--------------+-----------------------------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1
    """

    _methods = []
    _fields_of_interest = []
    _message = """
    NPR of type 1: Gravity non-specific non-physicality detected.
    """
    _scope = "1"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, self._scope, obj)

    @classmethod
    def _identify(cls, model):
        return []


class Type1aNPR(Type1NPR):
    r"""
    Non-Physical Region corresponding to regions of the cluster where the growth rates of the gas temperature and density are inconsistent
    with a positive gravitational field.

    +--------------+-----------------------------+
    | Properties                                 |
    +==============+=============================+
    | Scope        | Type 1a                     |
    +--------------+-----------------------------+
    | Methods      | :math:`\rho_g + T_g`        |
    +--------------+-----------------------------+
    | Gravity      | All                         |
    +--------------+-----------------------------+
    | Correctable? | True                        |
    +--------------+-----------------------------+

    **Class Diagram**

    .. inheritance-diagram:: cluster_generator.correction
        :parts: 1

    Notes
    -----
    When initializing a cluster from :math:`\rho_g` and :math:`T_g`, the gravitational field :math:`\nabla \Phi` may be determined by

    .. math::

        \nabla \Phi = \frac{-k_bT}{m_p\eta}\left[\partial_r \ln(\rho_g) + \partial_r \ln(T_g)\right]

    Because the cluster is assumed to be spherically symmetric, it should have a gravitational field which is determined entirely from the mass within
    any spherical shell and should therefore always be positive. Thus,

    .. math::

        \frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} < 0.

    This is the condition checked by Type1a non-physical regions.

    .. note::

        This method relies only on field values and does not consider the profiles available even if they exist.

    """
    _methods = ["from_dens_and_temp"]
    _message = """
    NPR of type 1a: Inconsistent temperature / density slopes.
    """
    _scope = "1a"
    correctable = True

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, obj)

    @classmethod
    def _identify(cls, model):
        nprs = []  # The NPRs being returned

        if "method" not in model.properties["meth"]:
            mylog.warning(
                "Failed to identify a 'generation_type' for the model. Unable to check for NPRs."
            )
        elif model.properties["meth"]["method"] not in cls._methods:
            return nprs
        else:
            pass

        # -- Main identification region -- #
        if "gravitational_field" in model.fields:
            holes = _check_non_positive(
                -model.fields["gravitational_field"].d, domain=model.fields["radius"].d
            )
        else:
            return nprs

        if len(holes) != 0:
            for d in holes:
                nprs.append(cls(d[0], d[1], model))

        if len(nprs) != 0:
            mylog.warning(
                f"Located {len(nprs)} non-physicalities of type {cls._scope}."
            )
        return nprs

    def _correct(self, correction_parameter=0.8):
        from scipy.interpolate import InterpolatedUnivariateSpline
        from unyt import unyt_array

        from cluster_generator.numalgs import positive_interpolation, solve_temperature

        # Correcting the non-positive regime
        rr, gf = self.obj["radius"].d, -self.obj["gravitational_field"].d

        nrr, ngf = positive_interpolation(
            rr, gf, correction_parameter=correction_parameter
        )

        new_field = unyt_array(-1 * ngf, self.obj["gravitational_field"].units)

        # Recalculating
        temperature = solve_temperature(
            self.obj["radius"], new_field, self.obj["density"]
        )

        dr = InterpolatedUnivariateSpline(nrr, self.obj["density"].d)
        tr = InterpolatedUnivariateSpline(nrr, temperature.d)

        if "stellar_density" in self.obj.properties["meth"]["profiles"]:
            stellar_density = self.obj.properties["meth"]["profiles"]["stellar_density"]
        else:
            stellar_density = None

        return ClusterModel.from_dens_and_temp(
            np.amin(nrr),
            np.amax(nrr),
            dr,
            tr,
            stellar_density=stellar_density,
            num_points=len(rr),
            **self.obj.properties,
        )


class CorrectionFailure(Exception):
    pass
