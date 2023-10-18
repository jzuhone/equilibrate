"""
Methods for correcting non-physicality and other issues in CGP's :py:class:`model.ClusterModel` instances.
"""
import numpy as np
from utils import mylog


class NonPhysicalRegion:
    """
    Marker class for non-physical behaviors in :py:class:`model.ClusterModel` instances.
    """

    _fields_of_interest = []
    _message = """
    Non-descript catch-all class of NPR. Progenitor class for all other NPR classes.
    """
    _scope = "generic"

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

    def resolve(self):
        """Attempts to resolve the non-physicality."""
        raise CorrectionFailure(
            f"There is no implemented resolution methodology for Type-{self._scope} NonPhysicalRegion objects: {self.__str__()}"
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

        # -- Subclass identification -- #
        if recursive and len(cls.__subclasses__()):
            mylog.info(
                f"Seeking NPRs in {len(cls.__subclasses__())} subclasses of {cls}"
            )
            for c in cls.__subclasses__():
                if hasattr(c, "identify"):
                    nprs += c.identify(model, recursive=recursive)

        return nprs


class Type0NPR(NonPhysicalRegion):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.NonPhysicalRegion`) which indicates NPRs of type 0.

    .. note::
        NPRs of type 0 are characterized by non-physicality inherent in the basic profiles of initialization. For example,
        a user trying to initialize with a negative temperature profile.
    """

    _fields_of_interest = []
    _message = """
    NPR of type 0: indicates that the user has erroneously specified a profile before any computation occurred.
    """
    _scope = "0"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, "0", obj)


class Type0aNPR(Type0NPR):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.Type0NPR`) which indicates NPRs of type 0a.

    .. note::
        NPRs of type 0a are characterized by non-physicality inherent in the basic profiles of initialization. In this case, negative temperature profiles.
    """

    _fields_of_interest = ["temperature"]
    _message = """
    NPR of type 0a: indicates that the user has erroneously specified a profile before any computation occurred. Specific to temperature profile.
    """
    _scope = "0a"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, "0a", obj)

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

        if "generation_type" not in model.parameters:
            mylog.warning(
                "Failed to identify a 'generation_type' for the model. Unable to check for NPRs."
            )
        elif model.parameters["generation_type"] != "from_dens_and_temp":
            pass
        else:
            for field in cls._fields_of_interest:
                if field in model.fields:
                    domains = _check_non_positive(
                        model.fields[field], domain=model.fields["radius"]
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
        # -- Subclass identification -- #
        if recursive and len(cls.__subclasses__()):
            mylog.info(
                f"Seeking NPRs in {len(cls.__subclasses__())} subclasses of {cls}"
            )
            for c in cls.__subclasses__():
                if hasattr(c, "identify"):
                    nprs += c.identify(model, recursive=recursive)

        return nprs


class Type0bNPR(Type0NPR):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.Type0NPR`) which indicates NPRs of type 0b.

    .. note::
        NPRs of type 0b are characterized by non-physicality inherent in the basic profiles of initialization. In this case, negative density (total or gaseous) profiles.
    """

    _fields_of_interest = ["density", "total_density"]
    _message = """
    NPR of type 0b: indicates that the user has erroneously specified a profile before any computation occurred. Specific to density profiles.
    """
    _scope = "0b"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, "0b", obj)

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

        if "generation_type" not in model.parameters:
            mylog.warning(
                "Failed to identify a 'generation_type' for the model. Unable to check for NPRs."
            )
        elif model.parameters["generation_type"] != "from_dens_and_tden":
            pass
        else:
            for field in cls._fields_of_interest:
                if field in model.fields:
                    domains = _check_non_positive(
                        model.fields[field], domain=model.fields["radius"]
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
        # -- Subclass identification -- #
        if recursive and len(cls.__subclasses__()):
            mylog.info(
                f"Seeking NPRs in {len(cls.__subclasses__())} subclasses of {cls}"
            )
            for c in cls.__subclasses__():
                if hasattr(c, "identify"):
                    nprs += c.identify(model, recursive=recursive)

        return nprs


class Type0cNPR(Type0NPR):
    """
    Non-Physical Region marker class (Subclass of :py:class:`correction.Type0NPR`) which indicates NPRs of type 0c.

    .. note::
        NPRs of type 0c are characterized by non-physicality inherent in the basic profiles of initialization. In this case, any initialization profile not
        considered in :py:class:`correction.Type0a` or :py:class:`correction.Type0b`.
    """

    _fields_of_interest = ["density", "total_density", "temperature", "entropy"]
    _message = """
    NPR of type 0: indicates that the user has erroneously specified a profile before any computation occurred. Not elsewhere specified.
    """
    _scope = "0c"

    def __init__(self, rmin, rmax, obj):
        super().__init__(rmin, rmax, "0c", obj)

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

        if "generation_type" not in model.parameters:
            mylog.warning(
                "Failed to identify a 'generation_type' for the model. Unable to check for NPRs."
            )
        elif model.parameters["generation_type"] in [
            "from_dens_and_temp",
            "from_dens_and_tden",
        ]:
            pass
        else:
            for field in cls._fields_of_interest:
                if field in model.fields:
                    domains = _check_non_positive(
                        model.fields[field], domain=model.fields["radius"]
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
        # -- Subclass identification -- #
        if recursive and len(cls.__subclasses__()):
            mylog.info(
                f"Seeking NPRs in {len(cls.__subclasses__())} subclasses of {cls}"
            )
            for c in cls.__subclasses__():
                if hasattr(c, "identify"):
                    nprs += c.identify(model, recursive=recursive)

        return nprs


class CorrectionFailure(Exception):
    pass


def identify_domain_borders(array, domain=None):
    """
    Identify the edges of the domains specified in the array.

    Parameters
    ----------
    array: np.ndarray
        Array (1D) containing ``1`` indicating truth and ``2`` indicating false from which to obtain the boundaries.
    domain: np.ndarray, optional
        The domain of consideration (x-values) to mark the boundaries instead of using indices.

    Returns
    -------
    list
        List of 2-tuples containing the boundary indices (if ``domain==None``) or the boundary positions if domain is specfied.
    """
    boundaries = (
        np.concatenate([[-1], array[:-1]]) + array + np.concatenate([array[1:], [-1]])
    )
    if domain is None:
        ind = np.indices(array).reshape((array.size,))
        vals = ind[np.where(boundaries == 1)]
    else:
        vals = domain[np.where(boundaries == 1)]

    return vals.reshape(len(vals) // 2, 2)


def _check_non_positive(array, domain=None):
    o = np.zeros(array.size)
    o[np.where(array < 0)] = 1
    return identify_domain_borders(o, domain=domain)
