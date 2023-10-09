"""
Utility functions for basic functionality of the py:module:`cluster_generator` package.
"""
import logging
import multiprocessing
import os
import pathlib as pt
import sys
import time
import warnings

import yaml
from more_itertools import always_iterable
from numpy.random import RandomState
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import physical_constants as pc
from unyt import unyt_array, unyt_quantity, kpc, Unit, exceptions

# -- configuration directory -- #
_config_directory = os.path.join(pt.Path(__file__).parents[0], "bin", "config.yaml")


# defining the custom yaml loader for unit-ed objects
def _yaml_unit_constructor(loader: yaml.FullLoader, node: yaml.nodes.MappingNode):
    kw = loader.construct_mapping(node)
    i_s = kw["input_scalar"]
    del kw["input_scalar"]
    return unyt_array(i_s, **kw)


def _yaml_lambda_loader(loader: yaml.FullLoader, node: yaml.nodes.ScalarNode):
    return eval(loader.construct_scalar(node))


def _get_loader():
    loader = yaml.FullLoader
    loader.add_constructor("!unyt", _yaml_unit_constructor)
    loader.add_constructor("!lambda", _yaml_lambda_loader)
    return loader


# -- loading the yaml configuration file -- #
try:
    with open(_config_directory, "r+") as config_file:
        cgparams = yaml.load(config_file, _get_loader())

except FileNotFoundError as er:
    raise FileNotFoundError(
        f"Couldn't find the configuration file! Is it at {_config_directory}? Error = {er.__repr__()}")
except yaml.YAMLError as er:
    raise yaml.YAMLError(f"The configuration file is corrupted! Error = {er.__repr__()}")

# warnings.filterwarnings("ignore")
stream = (sys.stdout if cgparams["system"]["logging"]["stream"] in ["STDOUT", "stdout"] else sys.stderr)
cgLogger = logging.getLogger("cluster_generator")

cg_sh = logging.StreamHandler(stream=stream)

# create formatter and add it to the handlers
formatter = logging.Formatter(cgparams["system"]["logging"]["ufstring"])
cg_sh.setFormatter(formatter)
# add the handler to the logger
cgLogger.addHandler(cg_sh)
cgLogger.setLevel(cgparams["system"]["logging"]["level"])
cgLogger.propagate = False

mylog = cgLogger

# -- Setting up the developer debugger -- #
devLogger = logging.getLogger("development_logger")

if cgparams["system"]["logging"]["developer_log"]["is_enabled"]:  # --> We do want to use the development logger.
    # -- checking if the user has specified a directory -- #
    if cgparams["system"]["logging"]["developer_log"]["output_directory"] is not None:
        from datetime import datetime

        dv_fh = logging.FileHandler(os.path.join(cgparams["system"]["logging"]["developer_log"]["output_directory"],
                                                 f"{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}.log"))

        # adding the formatter
        dv_formatter = logging.Formatter(cgparams["system"]["logging"]["developer_log"]["format"])

        dv_fh.setFormatter(dv_formatter)
        devLogger.addHandler(dv_fh)
        devLogger.setLevel(cgparams["system"]["logging"]["developer_log"]["level"])
        devLogger.propagate = False

    else:
        mylog.warning(
            "User enabled development logger but did not specify output directory. Dev logger will not be used.")
else:
    devLogger.propagate = False
    devLogger.disabled = True


def log_string(message):
    return cgparams["system"]["logging"]["ufstring"] % {"name"   : "cluster_generator",
                                                        "asctime": time.strftime("%Y-%d-%b %H:%M:%S", time.localtime()),
                                                        "message": message, "levelname": "INFO"}


def eprint(message, n, location=None, frmt=True, **kwargs):
    if cgparams["system"]["logging"]["level"] in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
        if frmt:
            print("%(tabs)s[%(location)s]: %(asctime)s %(message)s" % {
                "location": (location if location is not None else "-"),
                "asctime" : time.strftime("%Y-%d-%b %H:%M:%S", time.localtime()), "message": message, "tabs": "\t" * n},
                  file=stream, **kwargs)
        else:
            print(message, file=stream, **kwargs)


#: Proton Mass in ``Msun``.
mp = (pc.mp).to("Msun")
#: Newtons constant in ``kpc**3/Msun/Myr**2``.
G = (pc.G).to("kpc**3/Msun/Myr**2")
#: Boltzmann Constant in ``Msun*kpc**2/Myr**2/K``.
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
#: 1 kpc in centimeters.
kpc_to_cm = (1.0 * kpc).to_value("cm")

#: Hydrogen abundance
X_H = cgparams["physics"]["hydrogen_abundance"]

#: mean molecular mass
mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))

# -- Utility functions -- #
_truncator_function = lambda a, r, x: 1 / (1 + (x / r) ** a)


class TimeoutException(Exception):
    def __init__(self, msg='', func=None, max_time=None):
        self.msg = f"{msg} -- {str(func)} -- max_time={max_time} s"


def _daemon_process_runner(*args, **kwargs):
    # Runs the function specified in the kwargs in a daemon process #

    send_end = kwargs.pop("__send_end")
    function = kwargs.pop("__function")

    try:
        result = function(*args, **kwargs)
    except Exception as e:
        send_end.send(e)
        return

    send_end.send(result)


def time_limit(function, max_execution_time, *args, **kwargs):
    import time
    from tqdm import tqdm

    recv_end, send_end = multiprocessing.Pipe(False)
    kwargs["__send_end"] = send_end
    kwargs["__function"] = function

    tqdm_kwargs = {}
    for key in ["desc"]:
        if key in kwargs:
            tqdm_kwargs[key] = kwargs.pop(key)

    N = 1000

    p = multiprocessing.Process(target=_daemon_process_runner, args=args, kwargs=kwargs)
    p.start()

    for n in tqdm(range(N), **tqdm_kwargs,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining} - {postfix}]', colour="green",
                  leave=False):
        time.sleep(max_execution_time / 1000)

        if not p.is_alive():
            p.join()
            result = recv_end.recv()
            break

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutException("Failed to complete process within time limit.", func=function,
                               max_time=max_execution_time)
    else:
        p.join()
        result = recv_end.recv()

    if isinstance(result, Exception):
        raise result
    else:
        return result


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


def truncate_spline(f, r_t, a):
    """
    Takes the function ``f`` and returns a truncated equivalent of it, which becomes

    .. math::

    f'(x) = f(r_t) \left(\frac{x}{r_t}\right)**(r_t*df/dx(r_t)/f(r_t))

    This preserves the slope and continuity of the function be yields a monotonic power law at large :math:`r`.
    Parameters
    ----------
    f: InterpolatedUnivariateSpline
        The function to truncate
    r_t: float
        The scale radius
    a: float
        Truncation rate. Higher values cause transition more quickly about :math:`r_t`.

    Returns
    -------
    callable
        The new function.

    Examples
    --------

    .. code_block:: python

        from cluster_generator.radial_profiles import hernquist_density_profile
        from scipy.interpolate import InterpolatedUnivariateSpline
        import matplotlib.pyplot as plt
        x = np.geomspace(0.1,1000,1000)
        rho = hernquist_density_profile(1e6,1000)(x)
        rho_spline = InterpolatedUnivariateSpline(x,rho)
        xl = np.geomspace(0.1,1e7,1000)
        _rho_trunc = truncate_spline(rho_spline,1000,7)
        plt.figure()
        plt.loglog(x,rho,"k-",lw=3)
        plt.loglog(xl,rho_spline(xl),"k:")
        plt.loglog(xl,_rho_trunc(xl),"r-.")
        plt.show()

    """
    _gamma = r_t * f(r_t, 1) / f(r_t)  # This is the slope.
    return lambda x, g=_gamma, a=a, r=r_t: f(x) * _truncator_function(a, r, x) + (1 - _truncator_function(a, r, x)) * (
            f(r) * _truncator_function(-g, r, x))

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
    mass_int = lambda r: profile(r) * r * r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4. * np.pi * quad(mass_int, 0, r)[0]
    return mass


def integrate(profile, rr, rmax=None):
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
    if rmax is None:
        rmax = rr[-1]

    ret = np.zeros(rr.shape)
    with warnings.catch_warnings(record=True) as w:
        for i, r in enumerate(rr):
            ret[i] = quad(profile, r, rmax)[0]

        errs = w

    return ret, errs


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


def moving_average(array, n):
    return np.convolve(array, np.ones(n), "same") / n


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

    prng = parse_prng(prng)

    # - finding index for the ``r_max`` value. Allows us to renormalize by the mass at r_max.
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx - 1]  # Resampling the total mass

    u = prng.uniform(size=num_particles)

    # - Building the cumulative distribution - #
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)

    # - Inversely sampling the distribution at points ``u`` from x=P_r, y=r.
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot


def build_yt_dataset_fields(grid, models, domain_dimensions, centers, velocities):
    from cluster_generator.model import ClusterModel

    # -- Segmenting the fields by type -- #
    _added_fields = ["density", "pressure", "dark_matter_density",
                     "stellar_density", "gravitational_potential"]
    _mass_weighted_fields = ["temperature"]
    _mass_weighted_nonfields = ["velocity_x", "velocity_y", "velocity_z"]
    units = {
        "density"                : "Msun/kpc**3",
        "pressure"               : "Msun/kpc/Myr**2",
        "dark_matter_density"    : "Msun/kpc**3",
        "stellar_density"        : "Msun/kpc**3",
        "temperature"            : "K",
        "gravitational_potential": "kpc**2/Myr**2",
        "velocity_x"             : "kpc/Myr",
        "velocity_y"             : "kpc/Myr",
        "velocity_z"             : "kpc/Myr",
        "magnetic_field_strength": "G"
    }

    # -- Sanity Checks -- #
    models = ensure_list(models)

    for mid, model in enumerate(models):
        if isinstance(model, str):
            models[mid] = ClusterModel.from_h5_file(model)

    centers = ensure_ytarray(centers, "kpc")
    velocities = ensure_ytarray(velocities, "kpc/Myr")

    mylog.info("Building yt dataset structure...")

    x, y, z = grid
    fields = _added_fields + _mass_weighted_fields
    data = {}
    for i, p in enumerate(models):
        xx = x - centers.d[i][0]
        yy = y - centers.d[i][1]
        zz = z - centers.d[i][2]
        rr = np.sqrt(xx * xx + yy * yy + zz * zz)
        fd = InterpolatedUnivariateSpline(p["radius"].d,
                                          p["density"].d)

        # -- Managing constituent fields -- #
        for field in fields:
            if field not in p:  # We don't have this data, don't do anything.
                continue
            else:
                # Managing units
                try:
                    p[field] = p[field].to(units[field])
                except exceptions.UnitConversionError as error:
                    if field == "temperature":
                        p[field] = p[field].to("K", equivalence="thermal")
                    else:
                        raise error

            if field not in data:  # This data needs to be initialized in the data object.
                data[field] = unyt_array(
                    np.zeros(domain_dimensions), units[field]
                )

            f = InterpolatedUnivariateSpline(p["radius"].d,
                                             p[field].d)

            if field in _added_fields:
                data[field] += unyt_array(f(rr), units[field])  # Just add the values
            elif field in _mass_weighted_fields:
                data[field] += unyt_array(f(rr) * fd(rr), Unit(units[field]))  # Density weighted values

        # -- Managing Velocities -- #
        for j, field in enumerate(_mass_weighted_nonfields):
            if field not in data:
                data[field] = unyt_array(
                    np.zeros(domain_dimensions), units[field]
                )

            # We do need to add it #
            data[field] += unyt_array(velocities.d[i][j] * fd(rr), Unit(units[field]))

    if "density" in data:
        for field in _mass_weighted_fields + _mass_weighted_nonfields:
            data[field] /= data["density"].d
    else:
        mylog.warning("Failed to obtain a density profile, many fields may be inaccurate.")

    return data

def _find_holes(x,y,rtol=1e-3,dy=None):
    """
    locates holes in the domain of y on which the function is non-monotone.
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    _x,_y = x[:],y[:]
    if dy is None:
        secants = np.gradient(_y,_x)
    else:
        secants = dy(_x)

    holes = np.zeros(_x.size)
    ymax = np.maximum.accumulate(_y)
    holes[~np.isclose(_y,ymax,rtol=rtol)] = 1
    holes[secants<=-1e-8] = 1

    # construct boundaries of holes
    _hb = (np.concatenate([[holes[0]],holes])[:-1] +holes+ np.concatenate([holes,[holes[-1]]])[1:])
    ind = np.indices(_x.shape).reshape((_x.size,))

    hx,hy,hi = _x[np.where(_hb==1)],_y[np.where(_hb==1)],ind[np.where(_hb==1)]

    if holes[0] == 1:
        hx,hy,hi = np.concatenate([[_x[0]],hx]),np.concatenate([[hy[0]],hy]),np.concatenate([[0],hi])
    if holes[-1] == 1:
        hx,hy,hi = np.concatenate([hx,[_x[-1]]]),np.concatenate([hy,[hy[-1]]]),np.concatenate([hi,[ind[-1]]])

    plt.plot(_x,holes)
    plt.show()
    return len(hx)//2,np.array([hx.reshape(len(hx)//2,2),hy.reshape(len(hy)//2,2),hi.reshape(len(hi)//2,2)])

def monotone_interpolation(x,y,buffer=10,rtol=1e-3):
    from scipy.interpolate import CubicHermiteSpline
    if y[-1]>y[0]:
        monotonicity = 0
        _x,_y = x[:],y[:]
    elif y[0]>y[-1]:
        monotonicity = 1
        _x,_y = x[:],y[::-1]
    else:
        mylog.warning("Attempted to find holes in profile with no distinct monotonicity.")
        return None

    nholes,holes = _find_holes(_x,_y,rtol=rtol)
    derivatives = np.gradient(_y,_x,edge_order=2)
    plt.plot(_x,_y)
    plt.show()
    while nholes > 0:
        print(nholes)
        # carry out the interpolation over the hole.
        hxx,hyy,hii = holes[:,0,:]

        # building the interpolant information
        hii[1] = hii[1] + np.min(np.concatenate([[buffer,len(_x)-1-hii[1]],(holes[2,1:,0]-hii[1]).ravel()]))
        hii = np.array(hii,dtype="int")
        hyy = [_y[hii[0]],np.amax([_y[hii[1]],_y[hii[0]]])]
        hxx = [_x[hii[0]],_x[hii[1]]]

        if hii[1] == len(_x)-1:
            print(np.amax(_y))
            _y[hii[0]:hii[1]+1] = _y[hii[0]]
            print(_y[-10:],hxx,hyy,hii)
            input()
        else:
            xb,yb = hxx[1] - (hyy[1]-hyy[0])/(2*derivatives[hii[1]]), (1/2)*(hyy[0]+hyy[-1])
            s = [(yb-hyy[0])/(xb-hxx[0]),(hyy[1]-yb)/(hxx[1]-xb)]
            p = (s[0]*(hxx[1]-xb)+(s[1]*(xb-hxx[0])))/(hxx[1]-hxx[0])
            xs = [hxx[0],xb,_x[hii[-1]]]
            ys = [hyy[0],yb,_y[hii[-1]]]
            dys = [0.0,np.amin([2*s[0],2*s[1],p]),derivatives[hii[1]]]

            cinterp = CubicHermiteSpline(xs,ys,dys)
            _y[hii[0]:hii[1]] = cinterp(_x[hii[0]:hii[1]])

        plt.plot(_x,_y)
        plt.show()
        nholes, holes = _find_holes(_x, _y, rtol=rtol)

    if monotonicity == 1:
        _x,_y = _x[:],_y[::-1]

    return _x,_y




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(1, 10, 1000)
    y = 2 * np.cos(3 * x) - 2 * x
    dy = 2 - 6 * np.sin(3 * x)
    plt.plot(x,y)
    plt.show()
    nx,ny = monotone_interpolation(x,y)
    plt.plot(nx,ny)
    plt.show()