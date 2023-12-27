"""
Backend module for managing interfacing with :py:mod:`yt` and other external libraries / data structures.
"""
import gc
import os

import h5py
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm import tqdm

from cluster_generator.utils import _build_chunks, cgparams, chunked_operation, mylog

#: Field names corresponding to cases where fields may be added without concern for a weight function.
additive_fields = [
    "density",
    "pressure",
    "dark_matter_density",
    "stellar_density",
    "gravitational_potential",
]
#: Fields which need to be weighted by gas density when models are being combined.
dens_weighted_fields = ["temperature"]  # Density weighted fields.
#: density weighted non-fields
dens_weighted_nonfields = ["velocity_x", "velocity_y", "velocity_z"]
#: Field standard units.
units = {
    "density": ("Msun/kpc**3", None),
    "pressure": ("Msun/kpc/Myr**2", None),
    "dark_matter_density": ("Msun/kpc**3", None),
    "stellar_density": ("Msun/kpc**3", None),
    "temperature": ("K", "thermal"),
    "gravitational_potential": ("kpc**2/Myr**2", None),
    "velocity_x": ("kpc/Myr", None),
    "velocity_y": ("kpc/Myr", None),
    "velocity_z": ("kpc/Myr", None),
    "magnetic_field_strength": ("G", None),
}
_allowed_fields = additive_fields + dens_weighted_fields  # fields that are recognized.


def setup_yt_hdf5_file(
    fields, domain_dimensions, filename=None, overwrite=False, root="grid"
):
    """
    Set up an HDF5 file with correctly sized data grids for producing a fully realized :py:mod:`yt` dataset. This function
    will generate blank grids in the correct format for each of the specified fields provided as well as any additional meta-data
    necessary for :py:mod:`yt` to read the dataset.

    Parameters
    ----------
    fields: list of str
        The names of the relevant fields.

        .. warning::

            All provided fields will receive grids in the dataset; however, only recognized fields will actually have data
            written to them during the dataset generation process. A warning will be raised if an invalid field name is included
            in the dataset.

    domain_dimensions: tuple or list
        The size of each of the domains of the grids. These should be a length 3 tuple, list, or other iterable object which contain
        the number of cells to place in the grid along each of the data axes.
    filename: str, optional
        The filename at which to produce the HDF5 file. If ``filename`` is left as ``None`` (Default), then the HDF5 file is generated
        in a temporary directory within the systems ``/tmp`` directory (or OS equivalent). This file will be deleted after runtime has
        terminated.
    overwrite: bool, optional
        If ``True``, any existing HDF5 data at the ``filename`` location will be deleted and overwritten. Default is ``False``.
    root: str, optional
        The name of the ``HDF5`` "directory" at which to place the true datagrids. By default, this is ``grid``, which is what
        :py:mod:`yt` will look for when seeking readable data in HDF5.

    Returns
    -------
    str
        The filename for the output file.

    Raises
    ------
    IOError
        If the file cannot be generated or the parent directories don't exist.
    """
    import tempfile

    if filename is None:  # Use a tempfile to store the HDF5 data.
        tf = tempfile.NamedTemporaryFile(delete=False)
        fname = tf.name
        fo = h5py.File(tf, "a")
    else:
        if os.path.exists(filename) and not overwrite:
            raise IOError(
                f"The yt HDF5 file {filename} already exists and overwrite = False."
            )
        elif os.path.exists(filename):
            mylog.info(
                f"An existing HDF5 file was found at {filename}. Overwrite=True, so the data is being deleted."
            )
            os.remove(filename)
        else:
            pass

        try:
            fo = h5py.File(filename, "a")
        except FileNotFoundError:
            raise IOError(
                f"The parent directories of {filename} don't exist. HDF5 file cannot be generated."
            )

        fname = filename

    grid_group = fo.create_group(root)  # --> stores the true data
    geometry_group = fo.create_group(
        "geometry"
    )  # --> Stores the geometry of the space when too big for RAM.

    for field in fields:
        grid_group.create_dataset(field, domain_dimensions, dtype="float64")

    geometry_group.create_dataset("rr", domain_dimensions, dtype="float64")
    fo.close()

    mylog.info(f"Generated YT-HDF5 file at {fname}.")
    return fname


@chunked_operation
def _fill_grid(vin, field_function, weight_function=None):
    if weight_function is None:
        # fix the weight function to 1 if none is given.
        weight_function = lambda r: 1

    return weight_function(vin) * field_function(vin)


def setup_geometry(
    io_array,
    box_size,
    left_edge,
    domain_dimensions,
    chunking=True,
    chunksize=64,
    center=(0, 0, 0),
):
    """
    Construct the radial coordinate array for a given grid.

    Parameters
    ----------
    io_array: :py:class:`np.ndarray`
        The array into which the output data should be written.
    box_size: tuple or list
        A length 3 iterable with the values of each of the box sizes.
    left_edge: tuple or list
        A lenth 3 iterable containing the coordinates of the bottom left most point on the grid.
    domain_dimensions: tuple or list
        The number of "cells" to place on each axis of the grid.
    chunking: bool, optional
        (Default, ``True``) If enabled, chunking is used to reduce the memory load of the computation.
    chunksize: int, optional
        (Default, ``64``) The maximum size of a given chunk.
    center: tuple, optional
        (Default, ``(0,0,0)``) The center point of the chosen coordinate system.

    Returns
    -------
    :py:class:`np.ndarray`
        The boundary box for the output geometry. The geometry array itself is written to the ``io_array`` and not returned.
    """
    mylog.info(
        f"Constructing grid geometry on domain {domain_dimensions}, chunking = {chunking}"
    )
    domain_dimensions = np.array(domain_dimensions)
    if chunking:
        exp_mem = 1240 * np.prod(domain_dimensions / chunksize) / (8 * 1e6)
        mylog.info(
            f"Expected memory usage / iteration is {np.round(exp_mem,decimals=2)} MB."
        )
    else:
        exp_mem = 256 * np.prod(domain_dimensions) / (8 * 1e6)
        mylog.info(f"Expected memory usage is {np.round(exp_mem,decimals=2)} MB.")

    bbox = np.array(
        [
            [left_edge[0], left_edge[0] + box_size],
            [left_edge[1], left_edge[1] + box_size],
            [left_edge[2], left_edge[2] + box_size],
        ]
    )

    if chunking:
        # compute iteratively.
        _chunk_ids, _chunk_coords = _build_chunks(chunksize, domain_dimensions)

        for chunk in tqdm(
            _chunk_ids,
            desc="Constructing coordinate grids...",
            leave=False,
            disable=(not cgparams["system"]["display"]["progress_bars"]),
        ):
            _cc = _chunk_coords[:, :, *chunk]
            x, y, z = np.mgrid[
                _cc[0, 0] : _cc[1, 0], _cc[0, 1] : _cc[1, 1], _cc[0, 2] : _cc[1, 2]
            ] * ((bbox[:, 1] - bbox[:, 0]) / (domain_dimensions - 1)).reshape(
                3, 1, 1, 1
            ) + bbox[
                :, 0
            ].reshape(
                3, 1, 1, 1
            )
            rr = np.sqrt(
                (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
            )

            io_array[
                _cc[0, 0] : _cc[1, 0], _cc[0, 1] : _cc[1, 1], _cc[0, 2] : _cc[1, 2]
            ] = rr

        del x, y, z, rr
        gc.collect()
        return bbox
    else:
        # compute in a single pass.
        x, y, z = np.mgrid[
            bbox[0][0] : bbox[0][1] : domain_dimensions[0] * 1j,
            bbox[1][0] : bbox[1][1] : domain_dimensions[1] * 1j,
            bbox[2][0] : bbox[2][1] : domain_dimensions[2] * 1j,
        ]
        io_array = np.sqrt(x**2 + y**2 + z**2)

        return bbox


def compute_model_grids(model, geometry, io, chunking, chunksize, fields):
    """
    Computes the relevant grids for a specified :py:class:`model.ClusterModel` object.

    Parameters
    ----------
    model: py:class:`model.ClusterModel`
        The model from which to produce the grid data.
    geometry: :py:class:`np.ndarray`
        The geometry array on which the ``model`` is evaluated. This should be an array representing the grid in space with each value providing
        the radial position of the point.
    io: dict
        The IO object onto which the data should be written. If this is a ``dict``, then the data is written to an in-memory dictionary;
        however, if it is a pointer to an HDF5 file, then the data will be written to disk.
    chunking: bool
        If ``True``, the computations are broken down into chunks on each array to reduce the overall memory load of the algorithm.
    chunksize: int
        The maximum size of any given chunk.
    fields: list
        The list of fields from the model which are to be included in the dataset.

    Returns
    -------
    io
    """
    _fields = [
        fi for fi in fields if fi in (_allowed_fields) and (fi in model.fields)
    ]  # fields recognized, in model, and in list.

    if any(fi not in _fields for fi in fields):
        mylog.warning(
            f"Some fields are not written to YT dataset because they are not recognized or are not in the model: {[fi for fi in fields if fi not in _fields]}"
        )

    rr = model.fields["radius"].to_value("kpc")
    if "density" in model.fields:
        # construct a density spline for a weight function when there is gas.
        fd = InterpolatedUnivariateSpline(
            rr, model.fields["density"].to_value("Msun/kpc**3")
        )
    else:
        fd = lambda x: 1  # --> This is only relevant in DM only cases.

    for field in tqdm(
        _fields,
        desc=f"Constructing YT grids for {len(_fields)} fields.",
        leave=False,
        position=0,
        disable=(not cgparams["system"]["display"]["progress_bars"]),
    ):
        f_arr = model.fields[field].to_value(
            units[field][0], equivalence=units[field][1]
        )
        field_function = InterpolatedUnivariateSpline(rr, f_arr)

        if field in dens_weighted_fields:
            wf = fd
        else:
            wf = None

        _cd = _build_chunks(chunksize, domain_dimensions=geometry.shape)

        _fill_grid(
            [geometry],
            io[field],
            field_function,
            chunking=chunking,
            chunksize=chunksize,
            chunk_data=_cd,
            weight_function=wf,
            leave_progress_bar=False,
            progress_bar_position=1,
            show_progress_bar=cgparams["system"]["display"]["progress_bars"],
            label=field,
        )

    return io
