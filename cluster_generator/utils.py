"""
Utility functions for basic functionality of the py:module:`cluster_generator` package.
"""

import logging
import operator
import os
import pathlib as pt
import sys
from functools import reduce
from numbers import Number
from typing import Any, Callable, Collection, Iterable, Mapping, Self

import numpy as np
import ruamel.yaml
from more_itertools import always_iterable
from numpy.random import RandomState
from numpy.typing import ArrayLike
from scipy.integrate import quad
from unyt import Unit, kpc
from unyt import physical_constants as pc
from unyt import unyt_array, unyt_quantity

config_directory = os.path.join(pt.Path(__file__).parents[0], "bin", "config.yaml")
# :py:class:`pathlib.Path`: The directory in which the ``cluster_generator`` configuration files are located.

# Configure the ruamel.yaml environment to allow custom / 3rd party datatypes in the configuration yaml.
yaml = ruamel.yaml.YAML()
yaml.register_class(unyt_array)
yaml.register_class(unyt_quantity)
yaml.register_class(Unit)


class AttrDict(dict):
    """
    Attribute accessible dictionary.
    """

    def __init__(self, mapping: Mapping):
        super(AttrDict, self).__init__(mapping)
        self.__dict__ = self

        for key in self.keys():
            self[key] = self.__class__.from_nested_dict(self[key])

    @classmethod
    def from_nested_dict(cls, data: Any) -> Self:
        """Construct nested AttrDicts from nested dictionaries."""
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: cls.from_nested_dict(data[key]) for key in data})

    @classmethod
    def clean_types(cls, _mapping):
        for k, v in _mapping.items():
            if isinstance(v, AttrDict):
                _mapping[k] = cls.clean_types(_mapping[k])
            else:
                pass
        return dict(_mapping)

    def clean(self):
        return self.clean_types(self)


class YAMLConfiguration:
    """
    General class representing a YAML configuration file.
    """

    def __init__(self, path: pt.Path | str):
        self.path: pt.Path = pt.Path(path)
        # :py:class:`pathlib.Path`: The path to the underlying yaml file.
        self._config: ruamel.yaml.CommentedMap | None = None

    @property
    def config(self):
        if self._config is None:
            self._config = self.load()

        return AttrDict(self._config)

    @classmethod
    def load_from_path(cls, path: pt.Path) -> dict:
        """Read the configuration dictionary from disk."""
        try:
            with open(path, "r+") as cf:
                return yaml.load(cf)

        except FileNotFoundError as er:
            raise FileNotFoundError(
                f"Couldn't find the configuration file! Is it at {config_directory}? Error = {er.__repr__()}"
            )

    def load(self) -> dict:
        return self.__class__.load_from_path(self.path)

    def reload(self):
        """Reload the configuration file from disk."""
        self._config = None

    @classmethod
    def set_on_disk(cls, path: pt.Path | str, name: str | Collection[str], value: Any):
        _old = cls.load_from_path(path)

        if isinstance(name, str):
            name = name.split(".")
        else:
            pass

        setInDict(_old, name, value)

        with open(path, "w") as cf:
            yaml.dump(_old, cf)

    def set_param(self, name: str | Collection[str], value):
        self.__class__.set_on_disk(self.path, name, value)


cgparams: YAMLConfiguration = YAMLConfiguration(config_directory)
# :py:class:`YAMLConfiguration`: The configuration variable for ``cluster_generator``.


# Setting up the logging system
streams = dict(
    mylog=getattr(sys, cgparams.config.logging.mylog.stream),
    devlog=getattr(sys, cgparams.config.logging.devlog.stream),
)
_loggers = dict(
    mylog=logging.Logger("cluster_generator"), devlog=logging.Logger("CG-DEV")
)

_handlers = {}

for k, v in _loggers.items():
    # Construct the formatter string.
    _handlers[k] = logging.StreamHandler(streams[k])
    _handlers[k].setFormatter(
        logging.Formatter(getattr(cgparams.config.logging, k).format)
    )
    v.addHandler(_handlers[k])
    v.setLevel(getattr(cgparams.config.logging, k).level)
    v.propagate = False

    if k != "mylog":
        v.disabled = not getattr(cgparams.config.logging, k).enabled

mylog: logging.Logger = _loggers["mylog"]
# :py:class:`logging.Logger`: The main logger for ``pyXMIP``.
devlog: logging.Logger = _loggers["devlog"]
# :py:class:`logging.Logger`: The development logger for ``pyXMIP``.


mp: unyt_quantity = (pc.mp).to("Msun")
#: :py:class:`unyt.unyt_quantity`: Proton mass in solar masses.
G: unyt_quantity = (pc.G).to("kpc**3/Msun/Myr**2")
#: :py:class:`unyt.unyt_quantity`: Newton's gravitational constant in galactic units.
kboltz: unyt_quantity = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
#: :py:class:`unyt.unyt_quantity`: Boltzmann's constant
kpc_to_cm: float = (1.0 * kpc).to_value("cm")
# float: The conversion of 1 kpc to centimeters.

X_H: float = cgparams.config.physics.hydrogen_abundance
""" float: The cosmological hydrogen abundance.

The adopted value for :math:`X_H` may be changed in the ``cluster_generator`` configuration. Default is 0.76.
"""
mu: float = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
r""" float: The mean molecular mass given the cosmological hydrogen abundance :math:`X_H` and ignoring metals.

.. math::

    \mu = \frac{1}{\sum_j (j+1)X_j/A_j}

"""
mue: float = 1.0 / (X_H + 0.5 * (1.0 - X_H))
r""" float: The mean molecular mass per free electron in a fully ionized primordial plasma

.. math::

    \mu_e = \frac{\rho}{m_p n_e} = \frac{1}{\sum_j j X_j/A_j}

"""


# -- General purpose utility functions -- #
def getFromDict(dataDict: Mapping, mapList: Iterable[slice]) -> Any:
    """
    Fetch an object from a nested dictionary using a list of keys.

    Parameters
    ----------
    dataDict: dict
        The data dictionary to search.
    mapList: list
        The list of keys to follow.

    Returns
    -------
    Any
        The output value.

    """
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict: Mapping, mapList: Iterable[slice], value: Any):
    """
    Set the value of an object from a nested dictionary using a list of keys.

    Parameters
    ----------
    dataDict: dict
        The data dictionary to search.
    mapList: list
        The list of keys to follow.
    value: Any
        The value to set the object to
    """
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


class Registry:
    """
    Registry utility class.
    """

    def __init__(self):
        self._mapping = AttrDict(
            {}
        )  # This is an empty attribute dict that contains the registry objects.

    def __getattr__(self, name: str) -> Any:
        try:
            super().__getattribute__(name)
        except AttributeError:
            return getattr(self._mapping, name).obj

    def __str__(self):
        return f"Registry[{len(self._mapping)} items]"

    def __repr__(self):
        return self.__str__()

    @property
    def meta(self):
        return self._mapping

    def register(self, name: str, obj: Any, overwrite: bool = False, **kwargs):
        """
        Register an entity in the registry.

        Parameters
        ----------
        name: str
            The name of the entity to register.
        obj: Any
            The object to register.
        overwrite: bool
            Allow the registration to overwrite existing entries.
        kwargs:
            Additional metadata to associate with the registered object.
        """
        from types import SimpleNamespace

        # Check that the overwriting is valid.
        if name in self._mapping:
            assert (
                overwrite
            ), f"Cannot set {name} in {self} because overwrite = False and it is already registered."

        self._mapping[name] = SimpleNamespace(obj=obj, **kwargs)

    def unregister(self, name: str):
        """
        Unregister an entity from the registry.

        Parameters
        ----------
        name: str
            The object to remove from the registry.
        """
        del self._mapping[name]

    def autoregister(self, **meta) -> Callable[[Callable], Callable]:
        """
        Decorator for registration of objects at interpretation time.

        Parameters
        ----------
        **kwargs:
            Additional meta-data to attach to the registry for the specified item.
        """

        def _decorator(function: Callable) -> Callable:
            self.register(function.__name__, function, **meta)

            return function

        return _decorator

    def keys(self) -> Iterable[Any]:
        return self._mapping.keys()

    def values(self) -> Iterable[Any]:
        return self._mapping.values()

    def items(self) -> Iterable[tuple[Any, Any]]:
        return self._mapping.items()


def enforce_style(func):
    """Enforces the mpl style."""
    import matplotlib.pyplot as plt

    def wrapper(*args, **kwargs):
        _rcp_copy = plt.rcParams.copy()

        for _k, _v in cgparams.config.plotting.defaults.items():
            plt.rcParams[_k] = _v

        out = func(*args, **kwargs)

        plt.rcParams = _rcp_copy
        del _rcp_copy

        return out

    return wrapper


def ensure_ytquantity(x: Number | unyt_quantity, default_units: Unit) -> unyt_quantity:
    """
    Ensure that an input ``x`` is a unit-ed quantity with the expected units.

    Parameters
    ----------
    x: Any
        The value to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_quantity
        The corresponding quantity with correct units.

    """
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


def ensure_ytarray(x: ArrayLike, default_units: Unit) -> unyt_array:
    """
    Ensure that an input ``x`` is a unit-ed array with the expected units.

    Parameters
    ----------
    x: Any
        The values to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_array
        The corresponding array with correct units.

    """
    if isinstance(x, unyt_array):
        return x.to(default_units)
    else:
        return unyt_array(x, default_units)


def parse_prng(prng: RandomState | int) -> RandomState:
    """Return a random state from either random state or integer."""
    if isinstance(prng, RandomState):
        return prng
    else:
        return RandomState(prng)


def ensure_list(x: Collection) -> list:
    """Convert generic iterable to list."""
    return list(always_iterable(x))


# -- Mathematical utilities -- #
def integrate_mass(profile: Callable[[Number], float], rr: ArrayLike) -> float:
    r"""
    For a function :math:`f(r)`, perform the integral

    .. math::

        \int_0^{r_{\rm{max}}} 4\pi r^2 f(r) dr

    in quadrature.

    Parameters
    ----------
    profile: Callable
        The function to integrate.
    rr: array_like
        The array of radial values at which to perform the integration.

    Returns
    -------
    float
        The output value of the quadrature calculation.
    """
    mass_int = lambda r: profile(r) * r * r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.0 * np.pi * quad(mass_int, 0, r)[0]
    return mass


def integrate(profile: Callable[[Number], float], rr: ArrayLike) -> float:
    r"""
    For a function :math:`f(r)`, perform the integral

    .. math::

        \int_0^{r_{\rm{max}}} f(r) dr

    in quadrature.

    Parameters
    ----------
    profile: Callable
        The function to integrate.
    rr: array_like
        The array of radial values at which to perform the integration.

    Returns
    -------
    float
        The output value of the quadrature calculation.
    """
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    return ret


def integrate_toinf(profile: Callable[[Number], float], rr: ArrayLike) -> float:
    r"""
    For a function :math:`f(r)`, perform the integral

    .. math::

        \int_0^{\infty} f(r) dr

    in quadrature.

    Parameters
    ----------
    profile: Callable
        The function to integrate.
    rr: array_like
        The array of radial values at which to perform the integration.

    Returns
    -------
    float
        The output value of the quadrature calculation.
    """
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    ret[:] += quad(profile, rmax, np.inf, limit=100)[0]
    return ret


def generate_particle_radii(
    r: ArrayLike,
    m: ArrayLike,
    num_particles: int,
    r_max=None,
    prng: RandomState | int = None,
) -> tuple[ArrayLike, float]:
    """
    Use inverse cumulative sampling to determine placement radii for particles given a cumulative mass profile.

    Parameters
    ----------
    r: array_like
        The abscissa for the provided cumulative mass profile. Should be a ``(N,)`` array where ``N`` is the length of
        the provided mass profile.
    m: array_like
        The values of the cumulative mass corresponding to each radial position in ``r``.
    num_particles: int
        The number of particles to generate.
    r_max: float, optional
        The maximum radius to allow particles to be generated at.
    prng: int or RandomState, optional
        The random state to use (if replicability is necessary).

    Returns
    -------
    radius: array_like
        The radii of each of the particles.
    mtot: float
        The total mass of the particles. Equivalent to the cumulative mass at the cutoff radius.

    """
    prng = parse_prng(prng)

    # Manage the truncation point. Cumulative mass needs renormalization.
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx - 1]
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)

    # Generate random sample from 0-1 for inv. cum. sampling.
    u = prng.uniform(size=num_particles)
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot
