import numpy as np
from scipy.integrate import quad
import logging
from more_itertools import always_iterable
from unyt import unyt_array, unyt_quantity
from unyt import physical_constants as pc


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

mp = (pc.mp).to("Msun")
G = (pc.G).to("kpc**3/Msun/Myr**2")
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")

X_H = 0.76
mu = 1.0/(2.0*X_H + 0.75*(1.0-X_H))
mue = 1.0/(X_H+0.5*(1.0-X_H))


def integrate_mass(profile, rr):
    mass_int = lambda r: profile(r)*r*r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.*np.pi*quad(mass_int, 0, r)[0]
    return mass


def integrate(profile, rr):
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    return ret


def integrate_toinf(profile, rr):
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    ret[:] += quad(profile, rmax, np.inf, limit=100)[0]
    return ret


def generate_particle_radii(r, m, num_particles, r_max=None):
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx-1]
    u = np.random.uniform(size=num_particles)
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot


def ensure_ytquantity(x, units):
    if not isinstance(x, unyt_quantity):
        x = unyt_quantity(x, units)
    return x.to(units)


def ensure_ytarray(arr, units):
    if not isinstance(arr, unyt_array):
        arr = unyt_array(arr, units)
    return arr.to(units)


ensure_list = lambda x: list(always_iterable(x))
