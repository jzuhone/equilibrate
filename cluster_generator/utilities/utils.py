"""Utility functions for basic functionality of the py:module:`cluster_generator`
package."""


from numbers import Number
from typing import Callable, Iterable

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from scipy.integrate import quad


# -- General purpose utility functions -- #
def reverse_dict(dictionary: dict) -> dict:
    """Reverse a dictionary."""
    return {v: k for k, v in dictionary.items()}


def iterate_pairs(iterable: Iterable) -> dict:
    """Given a generic iterable, convert it to key-value pairs.

    Parameters
    ----------
    iterable: Iterable
        The iterable.

    Returns
    -------
    dict
        The resulting key-value paired map.
    """
    for item in iterable:
        if isinstance(item, dict):
            yield from item.items()  # Unpack the dictionary
        elif isinstance(item, tuple) and len(item) == 2:
            yield item  # Yield the tuple as is
        elif isinstance(item, (list, tuple)):
            yield from iterate_pairs(
                item
            )  # Recursively yield pairs if it's a list or tuple
        else:
            raise ValueError(f"Unsupported argument type: {type(item)}")


def parse_prng(prng: RandomState | int) -> RandomState:
    """Return a random state from either random state or integer."""
    if isinstance(prng, RandomState):
        return prng
    else:
        return RandomState(prng)


# -- Mathematical utilities -- #
def integrate_mass(profile: Callable[[Number], float], rr: ArrayLike) -> float:
    r"""For a function :math:`f(r)`, perform the integral.

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
    r"""For a function :math:`f(r)`, perform the integral.

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
    r"""For a function :math:`f(r)`, perform the integral.

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
    """Use inverse cumulative sampling to determine placement radii for particles given
    a cumulative mass profile.

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
