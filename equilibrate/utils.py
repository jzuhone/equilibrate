import numpy as np
from yt.funcs import get_pbar, mylog
from yt.units.yt_array import YTQuantity, YTArray
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from yt import units

mp = units.mp.in_units("Msun")
G = units.G.in_units("kpc**3/Msun/Myr**2")

def integrate_mass(profile, rr):
    mass_int = lambda r: profile(r)*r*r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.*np.pi*quad(mass_int, 0, r)[0]
    return mass

def integrate_toinf(profile, rr):
    prof_int = lambda r: profile(r)
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(prof_int, r, rmax)[0]
    ret[:] += quad(prof_int, rmax, np.inf, limit=100)[0]
    return ret