"""
==========
Relations
==========
The functions in this module provide empirical relations between profile variables in a variety of cases.
"""
import numpy as np
import unyt as u
from .utils import mue


def f_gas(M500, hubble=0.7):
    """
    The relationship between M500 and f_gas = Mgas/Mtot
    within r500 from Vikhlinin, A., et al. 2009, ApJ,
    692, 1033 (https://ui.adsabs.harvard.edu/abs/2009ApJ...692.1033V/).

    Parameters
    ----------
    M500 : float
        The M500 of the cluster in units of Msun.
    hubble : float, optional
        The Hubble parameter in units of 100 km/s/Mpc.
        Default: 0.7

    Returns
    -------
    f_gas at r500
    """
    m = M500*1.0e-15/hubble
    return ((0.72/hubble)**1.5)*(0.125+0.037*np.log10(m))


def m_bcg(M500):
    """
    """
    x = np.log10(M500) - 14.5
    y = 0.39*x+12.15
    return 10**y


def m_sat(M500):
    """
    """
    x = np.log10(M500) - 14.5
    y = 0.87*x+12.42
    return 10**y


def r_bcg(r200):
    x = np.log10(r200) - 1.0
    y = 0.95*x-0.3
    return 10**y


def convert_ne_to_density(ne):
    ne = ne*u.cm**-3
    return ne.to_value("Msun/kpc**3", "number_density", mu=mue)
