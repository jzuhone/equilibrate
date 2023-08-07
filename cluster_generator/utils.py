"""
Core utilities for logging and computation throughout the library.
"""
import numpy as np
from scipy.integrate import quad
import logging
from more_itertools import always_iterable
from unyt import unyt_array, unyt_quantity, kpc
from unyt import physical_constants as pc
from numpy.random import RandomState


# -------------------------------------------------------------------------------------------------------------------- #
# Constructing the Logger ============================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
cgLogger = logging.getLogger("cluster_generator")

ufstring = "%(name)-3s : [%(levelname)-9s] %(asctime)s %(message)s"
cfstring = "%(name)-3s : [%(levelname)-18s] %(asctime)s %(message)s"

cg_sh = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter(ufstring)
cg_sh.setFormatter(formatter)
# add the handler to the logger
cgLogger.addHandler(cg_sh)
cgLogger.setLevel('INFO')
cgLogger.propagate = False

mylog = cgLogger

# -------------------------------------------------------------------------------------------------------------------- #
# Units and Constants ================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
#: Proton Mass in ``Msun``.
mp = (pc.mp).to("Msun")
#: Newtons constant in ``kpc**3/Msun/Myr**2``.
G = (pc.G).to("kpc**3/Msun/Myr**2")
#: Boltzmann Constant in ``Msun*kpc**2/Myr**2/K``.
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
#: 1 kpc in centimeters.
kpc_to_cm = (1.0*kpc).to_value("cm")

#: Hydrogen abundance
X_H = 0.76
#: mean molecular mass
mu = 1.0/(2.0*X_H + 0.75*(1.0-X_H))
mue = 1.0/(X_H+0.5*(1.0-X_H))

# -------------------------------------------------------------------------------------------------------------------- #
# Type Assertions ==================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def ensure_ytquantity(x, units):
    if not isinstance(x, unyt_quantity):
        x = unyt_quantity(x, units)
    return x.to(units)


def ensure_ytarray(arr, units):
    if not isinstance(arr, unyt_array):
        arr = unyt_array(arr, units)
    return arr.to(units)


def parse_prng(prng):
    if isinstance(prng, RandomState):
        return prng
    else:
        return RandomState(prng)


def ensure_list(x):
    return list(always_iterable(x))

# -------------------------------------------------------------------------------------------------------------------- #
# Math Utilities ===================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def integrate_mass(profile, rr):
    """
    Integrates the density profile ``profile`` cumulatively over the radial array ``rr``.
    Parameters
    ----------
    profile: callable
        The density profile.
    rr: array-like
        The ``array`` of radii at which to compute the integral mass profile.

    Returns
    -------
    mass: array-like
        The resultant mass array.

    Notes
    -----

    .. attention::

        This function may be costly if run over a large array because each integral is computed individually instead
        of by increment.


    """
    mass_int = lambda r: profile(r)*r*r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.*np.pi*quad(mass_int, 0, r)[0]
    return mass


def integrate(profile, rr):
    """
    Integrates the profile ``profile`` cumulatively over the radial array ``rr``.
    Parameters
    ----------
    profile: callable
        The profile.
    rr: array-like
        The ``array`` of radii at which to compute the integral mass profile.

    Returns
    -------
    array-like
        The resultant mass array.

    Notes
    -----

    .. attention::

        This function may be costly if run over a large array because each integral is computed individually instead
        of by increment.
    """
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    return ret


def integrate_toinf(profile, rr):
    """
    Integrates the profile ``profile`` cumulatively over the radial array ``rr`` and then to ``inf``.
    Parameters
    ----------
    profile: callable
        The profile.
    rr: array-like
        The ``array`` of radii at which to compute the integral mass profile.

    Returns
    -------
    array-like
        The resultant mass array.

    Notes
    -----

    .. attention::

        This function may be costly if run over a large array because each integral is computed individually instead
        of by increment.
    """
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    ret[:] += quad(profile, rmax, np.inf, limit=100)[0]
    return ret


def generate_particle_radii(r, m, num_particles, r_max=None, prng=None):
    r"""
    Generates an array of sampled radii for ``num_particles`` particles subject to the mass distribution defined by ``r`` and ``m``.

    Parameters
    ----------
    r: array-like
        The radii on which the mass profile is defined. ``len(r) == len(m)``.
    m: array-like
        The cumulative mass such that ``m[i]`` is the mass within ``r[i]``.
    num_particles: int
        The number of particle positions to generate.
    r_max: float or int, optional
        The maximum radius at which to generate particles.
    prng: numpy.random.RandomState
        The pseudo-random number generator if desired. (Will be generated on its own if not).

    Returns
    -------
    radii: array-like
        The radii of each of the particles to be included.

    Notes
    -----
    This function relies on inverse cumulative sampling. We first determine the fractional mass profile

    .. math::

        P_r = \frac{m}{m_{\mathrm{max}}},

    from which, the function :math:`P_r(r)` then represents the probability distribution of the particles. Finally,
    an array ``u`` is generated by uniformly sampling from the inverse distribution.
    """
    #  Setup
    # ----------------------------------------------------------------------------------------------------------------- #
    prng = parse_prng(prng)

    # - finding index for the ``r_max`` value. Allows us to renormalize by the mass at r_max.
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx-1] # Resampling the total mass

    #  Sampling
    # ----------------------------------------------------------------------------------------------------------------- #
    u = prng.uniform(size=num_particles)

    # - Building the cumulative distribution - #
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)

    # - Inversely sampling the distribution at points ``u`` from x=P_r, y=r.
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot


