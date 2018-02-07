import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from yt import units

mp = units.mp.in_units("Msun")
G = units.G.in_units("kpc**3/Msun/Myr**2")
kboltz = units.kboltz

def integrate_mass(profile, rr):
    mass_int = lambda r: profile(r)*r*r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.*np.pi*quad(mass_int, 0, r)[0]
    return mass

def integrate(profile, rr):
    prof_int = lambda r: profile(r)
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(prof_int, r, rmax, epsabs=1.0e-5, epsrel=1.0e-5)[0]
    return ret

def integrate_toinf(profile, rr):
    prof_int = lambda r: profile(r)
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(prof_int, r, rmax)[0]
    ret[:] += quad(prof_int, rmax, np.inf, limit=100)[0]
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
    r_spline = InterpolatedUnivariateSpline(P_r, r, k=3, ext=1)
    #radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    radius = r_spline(u)
    return radius, mtot