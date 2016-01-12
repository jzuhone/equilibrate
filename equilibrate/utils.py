import numpy as np
from yt.funcs import get_pbar, mylog
from yt.units.unit_object import Unit
from yt.units.yt_array import YTQuantity, YTArray
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
import yt.utilities.physical_constants as pc
from six import string_types

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

def logspace(xmin, xmax, num):
    x = np.logspace(np.log10(xmin), np.log10(xmax), num, endpoint=True)
    return x*xmin.uq

cgs_units = {"length":"cm",
             "mass":"g",
             "time":"s",
             "temperature":"K"}

def setup_units(input_units):
    units = {}
    if input_units is None:
        input_units = {}
    for dim, cgs_unit in cgs_units.items():
        if dim in input_units:
            val = input_units[dim]
            if isinstance(val, string_types):
                u = Unit(val)
            else:
                u = Unit("%g*%s" % (val[0], val[1]))
        else:
            u = Unit(cgs_unit)
        units[dim] = u
    units["density"] = units["mass"]/(units["length"]**3)
    units["velocity"] = units["length"]/units["time"]
    units["specific_energy"] = units["velocity"]*units["velocity"]
    units["energy"] = units["specific_energy"]*units["mass"]
    units["acceleration"] = units["velocity"]/units["time"]
    units["pressure"] = units["specific_energy"]*units["density"]
    units["G"] = units["density"]**-1/units["time"]**2
    units["kB"] = units["energy"]/units["temperature"]
    return units

def unitful_zeros(shape, units):
    return YTArray(np.zeros(shape), units)

class InterpolateSplineWithUnits(InterpolatedUnivariateSpline):
    def __init__(self, x, y, **kwargs):
        if hasattr(x, "units"):
            self.input_units = x.units
        else:
            self.input_units = "dimensionless"
        if hasattr(y, "units"):
            self.output_units = y.units
        else:
            self.output_units = "dimensionless"
        super(InterpolateSplineWithUnits, self).__init__(x, y, **kwargs)

    def __call__(self, x, nu=0):
        if str(self.input_units) != "dimensionless":
            xx = x.in_units(self.input_units)
        else:
            xx = x
        yy = super(InterpolateSplineWithUnits, self).__call__(xx, nu=nu)
        y_units = self.output_units
        if nu > 0 and str(self.input_units) != "dimensionless":
            deriv_units = YTQuantity(1.0, self.input_units).units**nu
            y_units /= deriv_units
        return YTArray(super(InterpolateSplineWithUnits, self).__call__(xx, nu=nu),
                       y_units)
