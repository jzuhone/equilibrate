"""
Utility functions for basic functionality of the py:module:`cluster_generator` package.
"""
import functools
import logging
import multiprocessing
import os
import pathlib as pt
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from more_itertools import always_iterable
from numpy.random import RandomState
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import Unit, exceptions, kpc
from unyt import physical_constants as pc
from unyt import unyt_array, unyt_quantity

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


try:
    with open(_config_directory, "r+") as config_file:
        cgparams = yaml.load(config_file, _get_loader())

except FileNotFoundError as er:
    raise FileNotFoundError(
        f"Couldn't find the configuration file! Is it at {_config_directory}? Error = {er.__repr__()}"
    )
except yaml.YAMLError as er:
    raise yaml.YAMLError(
        f"The configuration file is corrupted! Error = {er.__repr__()}"
    )


stream = (
    sys.stdout
    if cgparams["system"]["logging"]["main"]["stream"] in ["STDOUT", "stdout"]
    else sys.stderr
)
cgLogger = logging.getLogger("cluster_generator")

cg_sh = logging.StreamHandler(stream=stream)

# create formatter and add it to the handlers
formatter = logging.Formatter(cgparams["system"]["logging"]["main"]["format"])
cg_sh.setFormatter(formatter)
# add the handler to the logger
cgLogger.addHandler(cg_sh)
cgLogger.setLevel(cgparams["system"]["logging"]["main"]["level"])
cgLogger.propagate = False

mylog = cgLogger

# -- Setting up the developer debugger -- #
devLogger = logging.getLogger("development_logger")

if cgparams["system"]["logging"]["developer"][
    "enabled"
]:  # --> We do want to use the development logger.
    # -- checking if the user has specified a directory -- #
    if cgparams["system"]["logging"]["developer"]["output_directory"] is not None:
        from datetime import datetime

        dv_fh = logging.FileHandler(
            os.path.join(
                cgparams["system"]["logging"]["developer"]["output_directory"],
                f"{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}.log",
            )
        )

        # adding the formatter
        dv_formatter = logging.Formatter(
            cgparams["system"]["logging"]["main"]["format"]
        )

        dv_fh.setFormatter(dv_formatter)
        devLogger.addHandler(dv_fh)
        devLogger.setLevel("DEBUG")
        devLogger.propagate = False

    else:
        mylog.warning(
            "User enabled development logger but did not specify output directory. Dev logger will not be used."
        )
else:
    devLogger.propagate = False
    devLogger.disabled = True


class LogMute:
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        self.logger.disabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.disabled = False


def _enforce_style(func):
    """enforces the mpl style."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _rcp_copy = plt.rcParams.copy()

        for _k, _v in cgparams["plotting"]["defaults"].items():
            plt.rcParams[_k] = _v

        out = func(*args, **kwargs)

        plt.rcParams = _rcp_copy
        del _rcp_copy

        return out

    return wrapper


mp = (pc.mp).to("Msun")
G = (pc.G).to("kpc**3/Msun/Myr**2")
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
kpc_to_cm = (1.0 * kpc).to_value("cm")

X_H = cgparams["physics"]["hydrogen_abundance"]
mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))

# -- Utility functions -- #
_truncator_function = lambda a, r, x: 1 / (1 + (x / r) ** a)


class TimeoutException(Exception):
    def __init__(self, msg="", func=None, max_time=None):
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

    for _ in tqdm(
        range(N),
        **tqdm_kwargs,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining} - {postfix}]",
        colour="green",
        leave=False,
    ):
        time.sleep(max_execution_time / 1000)

        if not p.is_alive():
            p.join()
            result = recv_end.recv()
            break

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutException(
            "Failed to complete process within time limit.",
            func=function,
            max_time=max_execution_time,
        )
    else:
        p.join()
        result = recv_end.recv()

    if isinstance(result, Exception):
        raise result
    else:
        return result


def truncate_spline(f, r_t, a):
    r"""
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
    """
    _gamma = r_t * f(r_t, 1) / f(r_t)  # This is the slope.
    return lambda x, g=_gamma, a=a, r=r_t: f(x) * _truncator_function(a, r, x) + (
        1 - _truncator_function(a, r, x)
    ) * (f(r) * _truncator_function(-g, r, x))


def integrate_mass(profile, rr):
    mass_int = lambda r: profile(r) * r * r
    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.0 * np.pi * quad(mass_int, 0, r)[0]
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


def generate_particle_radii(r, m, num_particles, r_max=None, prng=None):
    prng = parse_prng(prng)
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx - 1]
    u = prng.uniform(size=num_particles)
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot


def build_yt_dataset_fields(grid, models, domain_dimensions, centers, velocities):
    from cluster_generator.model import ClusterModel

    # -- Segmenting the fields by type -- #
    _added_fields = [
        "density",
        "pressure",
        "dark_matter_density",
        "stellar_density",
        "gravitational_potential",
    ]
    _mass_weighted_fields = ["temperature"]
    _mass_weighted_nonfields = ["velocity_x", "velocity_y", "velocity_z"]
    units = {
        "density": "Msun/kpc**3",
        "pressure": "Msun/kpc/Myr**2",
        "dark_matter_density": "Msun/kpc**3",
        "stellar_density": "Msun/kpc**3",
        "temperature": "K",
        "gravitational_potential": "kpc**2/Myr**2",
        "velocity_x": "kpc/Myr",
        "velocity_y": "kpc/Myr",
        "velocity_z": "kpc/Myr",
        "magnetic_field_strength": "G",
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
        fd = InterpolatedUnivariateSpline(p["radius"].d, p["density"].d)

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

            if (
                field not in data
            ):  # This data needs to be initialized in the data object.
                data[field] = unyt_array(np.zeros(domain_dimensions), units[field])

            f = InterpolatedUnivariateSpline(p["radius"].d, p[field].d)

            if field in _added_fields:
                data[field] += unyt_array(f(rr), units[field])  # Just add the values
            elif field in _mass_weighted_fields:
                data[field] += unyt_array(
                    f(rr) * fd(rr), Unit(units[field])
                )  # Density weighted values

        # -- Managing Velocities -- #
        for j, field in enumerate(_mass_weighted_nonfields):
            if field not in data:
                data[field] = unyt_array(np.zeros(domain_dimensions), units[field])

            # We do need to add it #
            data[field] += unyt_array(velocities.d[i][j] * fd(rr), Unit(units[field]))

    if "density" in data:
        for field in _mass_weighted_fields + _mass_weighted_nonfields:
            data[field] /= data["density"].d
    else:
        mylog.warning(
            "Failed to obtain a density profile, many fields may be inaccurate."
        )

    return data


def ensure_ytquantity(x, default_units):
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


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
