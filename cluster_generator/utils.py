"""
Utility functions for basic functionality of the py:module:`cluster_generator` package.
"""
import logging
import operator
import os
import pathlib as pt
import sys
from functools import wraps
from itertools import product

import numpy as np
import yaml
from more_itertools import always_iterable
from numpy.random import RandomState
from scipy.integrate import quad
from tqdm import tqdm
from unyt import kpc
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


mp = (pc.mp).to("Msun")
G = (pc.G).to("kpc**3/Msun/Myr**2")
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
kpc_to_cm = (1.0 * kpc).to_value("cm")

X_H = cgparams["physics"]["hydrogen_abundance"]
mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))

# -- Utility functions -- #
_truncator_function = lambda a, r, x: 1 / (1 + (x / r) ** a)


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


def _build_chunks(chunksize, domain_dimensions):
    domain_dimensions = np.array(domain_dimensions, dtype="uint16")
    chunksize = np.array(3 * [chunksize], dtype="uint16")
    ngrids = np.array(
        np.ceil(domain_dimensions / chunksize), dtype="uint32"
    )  # count of grids on each dimension.
    _gu, _gv, _gw = np.mgrid[0 : ngrids[0], 0 : ngrids[1], 0 : ngrids[2]]

    # Construct the index space coordinates for each of the chunks.
    _chunk_coordinates = np.array(
        [[_gu, _gv, _gw], [_gu + 1, _gv + 1, _gw + 1]], dtype="uint"
    )
    _chunk_coordinates[:, :, _gu, _gv, _gw] *= chunksize.reshape((1, 3, 1, 1, 1))

    _chunk_coordinates[1, 0, -1, :, :] = domain_dimensions[0]
    _chunk_coordinates[1, 1, :, -1, :] = domain_dimensions[1]
    _chunk_coordinates[1, 2, :, :, -1] = domain_dimensions[2]

    chunk_ids = list(product(*[range(k) for k in ngrids]))

    return chunk_ids, _chunk_coordinates


def chunked_operation(function):
    _ops = {"+=": operator.iadd, "*=": operator.imul, "=": None}

    @wraps(function)
    def wrapper(
        arrays_in,
        array_out,
        *args,
        chunksize=64,
        chunking=True,
        chunk_data=None,
        label=None,
        progress_bar_position=0,
        show_progress_bar=True,
        leave_progress_bar=False,
        oper="+=",
        **kwargs,
    ):
        """
        Wraps ``function`` so that the operation acts on arrays in a chunked fashion. Generically, ``function`` must have
        a structure ``function(array1,array2,...,arg1,arg2,...,kwarg1,kwarg2)``. The new, wrapped, function will have a signature
        ``function([arrays_in],array_out,*args,...,**wrapper_kwargs,**kwargs)``. The operations will then be carried out in chunks and
        each chunked operation written directly to ``array_out``.

        Parameters
        ----------
        arrays_in: list of :py:class:`np.ndarray`
            The input arrays. For whichever ``function`` is chosen as the base, the correct number of array arguments must be
            contained in this list.
        array_out: :py:class:`np.ndarray` or array-like
            This is the output array, and therefore should be the object onto which the data is written. This could be an instance
            of :py:class:`np.ndarray` stored in memory (in which case, chunking is effectively useless), or (as is more common), a reference
            to an HDF5 dataset or similar IO protocol so that each chunk is written to disk and memory is preserved.
        *args
            Additional arguments which are not arrays (and therefore do not need to be chunked). These are passed directly to ``function``.
        chunksize: int, optional
            The maximum size of a given chunk in the operation. Smaller ``chunksize`` will reduce memory load but increase runtime. Default is 64.
        chunking: bool, optional
            If ``True``, chunking is used; otherwise, the operation is done as usual without the wrapper. Default is ``True``.
        chunk_data: tuple, optional
            Output from :py:func:`_build_chunks`. The ``chunk_ids`` and ``chunk_coordinates``. This kwarg is not necessary; however, if
            many operations are performed on the same array structure, computing the chunk data once and specifying it in the kwarg will reduce
            the runtime associated with repeated computation.
        label: str, optional
            The name of the process being executed. If not specified, the default string representation of the function is used.
        progress_bar_position: int, optional
            The position of the progress bar.
        show_progress_bar: bool, optional
            If ``False``, the progress bar is hidden.
        leave_progress_bar: bool, optional
            If ``True``, the progress bar will persist after the operation is complete.
        **kwargs
            Additional kwargs to pass to ``function``.

        Returns
        -------
        callable
            The output wrapper function
        """
        if label is None:
            label = function.__str__()
        domain_dimensions = array_out.shape

        if chunking:
            if not chunk_data:
                chunk_ids, _chunk_coordinates = _build_chunks(
                    chunksize, domain_dimensions
                )
            else:
                chunk_ids, _chunk_coordinates = chunk_data

            for chunk in tqdm(
                chunk_ids,
                desc=f"Computing {label} on {len(chunk_ids)} grids...",
                position=progress_bar_position,
                leave=leave_progress_bar,
                disable=(not show_progress_bar),
            ):
                _cc = _chunk_coordinates[:, :, *chunk]
                _slice = [
                    slice(_cc[0, 0], _cc[1, 0]),
                    slice(_cc[0, 1], _cc[1, 1]),
                    slice(_cc[0, 2], _cc[1, 2]),
                ]

                if _ops[oper] is not None:
                    array_out[*_slice] = _ops[oper](
                        array_out[*_slice],
                        function(*[i[*_slice] for i in arrays_in], *args, **kwargs),
                    )
                else:
                    array_out[*_slice] = function(
                        *[i[*_slice] for i in arrays_in], *args, **kwargs
                    )
        else:
            if _ops[oper] is not None:
                array_out = _ops[oper](array_out, function(*arrays_in, *args, **kwargs))
            else:
                array_out = function(*arrays_in, *args, **kwargs)

    return wrapper


if __name__ == "__main__":
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 200), np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)
    )
    rr = np.sqrt(x**2 + y**2 + z**2)

    u = np.ones(rr.shape)
    vp = np.zeros(rr.shape)

    chunked_operation(np.divide)(
        [u, np.sin(rr)], vp, chunksize=10, leave_progress_bar=True
    )

    print(vp)
