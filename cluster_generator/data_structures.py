"""
Backend module for managing interfacing with :py:mod:`yt` and other external libraries / data structures.
"""
import os
import tempfile

import h5py
import numpy as np
from scipy.interpolate import dfitpack
from tqdm import tqdm

from cluster_generator import ClusterModel
from cluster_generator.opt.structures import (
    construct_chunks,
    dump_field_to_hdf5,
    renormalize,
    unnormalize,
)
from cluster_generator.utils import cgparams, ensure_ytarray, mylog

# from cluster_generator.opt.structures import interpolate_from_tables
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
dens_weighted_nonfields = {
    "velocity_x": ("density", 0, "velocity"),
    "velocity_y": ("density", 1, "velocity"),
    "velocity_z": ("density", 2, "velocity"),
    "stellar_velocity_x": ("stellar_density", 0, "velocity"),
    "stellar_velocity_y": ("stellar_density", 1, "velocity"),
    "stellar_velocity_z": ("stellar_density", 2, "velocity"),
    "dm_velocity_x": ("dark_mater_density", 0, "velocity"),
    "dm_velocity_y": ("dark_mater_density", 1, "velocity"),
    "dm_velocity_z": ("dark_mater_density", 2, "velocity"),
    "mixing": ("density", None, "model_id"),
}

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
    "stellar_velocity_x": ("kpc/Myr", None),
    "stellar_velocity_y": ("kpc/Myr", None),
    "stellar_velocity_z": ("kpc/Myr", None),
    "dm_velocity_x": ("kpc/Myr", None),
    "dm_velocity_y": ("kpc/Myr", None),
    "dm_velocity_z": ("kpc/Myr", None),
    "mixing": ("", None),
}
_allowed_fields = (
    additive_fields + dens_weighted_fields + list(dens_weighted_nonfields.keys())
)  # fields that are recognized.


class YTHDF5:
    """
    Wrapper class for YT style HDF5 files. Used to manage the writing of :py:class:`model.ClusterModel` instances
    to YT datasets.
    """

    def __init__(
        self, domain_dimensions, bbox, chunksize=64, fields=None, filename=None
    ):
        """
        Initializes the :py:class:`YTHDF5` instance.

        Parameters
        ----------
        domain_dimensions: array-like
            The dimensions of the grid along each axis (3-tuple)
        bbox: array-like
            The bounding box of the grid. Should be of size ``(3,2)``.
        chunksize: int, optional
            The maximum size of a chunk. Chunking is used to conserve memory during computations. A higher ``chunksize`` will
            increase the memory usage but decrease computation time, a lower value will do the opposite. Default is ``64``.
        fields: list of str, optional
            The fields to include in the dataset. If ``None``, then all of the available fields are included.
        filename: str, optional
            The filename for the underlying file. If left as ``None``, a ``tmpfile`` is generated to hold the dataset.
            Otherwise, if a ``str`` instance is provided, then that file will be used. If a pre-existing file is specified,
            then it will either overwrite or raise an error depending on ``overwrite``.
        """
        self._cm = None
        self._dd = np.array(domain_dimensions, dtype="uint32")
        self.bbox = np.array(bbox, dtype="float64")
        self.chunksize = chunksize
        self.fields = fields if fields is not None else _allowed_fields
        self._buffer = None
        self._gridbuffer = None

        if filename is None:
            tf = tempfile.NamedTemporaryFile(delete=False)
            self.filename = tf.name

        else:
            self.filename = filename

    @classmethod
    def new(
        cls,
        domain_dimensions,
        bbox,
        chunksize=64,
        fields=None,
        filename=None,
        overwrite=False,
    ):
        """
        Initialize a new :py:class:`YTHDF5` instance given specific generation parameters.

        Parameters
        ----------
        domain_dimensions: array-like
            The dimensions of the grid along each axis (3-tuple)
        bbox: array-like
            The bounding box of the grid. Should be of size ``(3,2)``.
        chunksize: int, optional
            The maximum size of a chunk. Chunking is used to conserve memory during computations. A higher ``chunksize`` will
            increase the memory usage but decrease computation time, a lower value will do the opposite. Default is ``64``.
        fields: list of str, optional
            The fields to include in the dataset. If ``None``, then all of the available fields are included.
        filename: str, optional
            The filename for the underlying file. If left as ``None``, a ``tmpfile`` is generated to hold the dataset.
            Otherwise, if a ``str`` instance is provided, then that file will be used. If a pre-existing file is specified,
            then it will either overwrite or raise an error depending on ``overwrite``.
        overwrite: bool, optional
            If ``True``, then the file will be overwritten if it already exists. Default ``False``.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`
            The generated YT dataset.
        """
        if filename is not None and os.path.exists(filename):
            if overwrite is True:
                mylog.info(f"Clearning pre-existing data from {filename}...")
                os.remove(filename)
            else:
                pass

        h5 = cls(domain_dimensions, bbox, chunksize, fields=fields, filename=filename)
        h5._setup()
        return h5

    @classmethod
    def read(cls, filename):
        """
        Read an :py:class:`data_structures.YTHDF5` instance from the ``.h5`` file.

        Parameters
        ----------
        filename: str
            The filename to read. The filename must exist.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`
        """
        properties = {
            k: None for k in ["domain_dimensions", "bbox", "chunksize", "fields"]
        }
        assert os.path.exists(filename), f"The file {filename} doesn't appear to exist."

        fo = h5py.File(filename, "a")
        for k in properties:
            try:
                properties[k] = fo.attrs[k]
            except KeyError:
                raise SyntaxError(
                    f"The YTHDF5 file {filename} doesn't have the appropriate attributes. (FAILED TO FIND {k})."
                )
        properties["filename"] = filename

        o = cls(
            properties["domain_dimensions"],
            properties["bbox"],
            properties["chunksize"],
            fields=properties["fields"],
            filename=filename,
        )
        try:
            o._cm = fo["chunks"]["chunkmap"][...]
        except KeyError:
            raise IOError("Failed to locate the chunkmap for this file.")

        fo.close()
        return o

    def _ensure_open(self):
        if self._buffer is None or not self._buffer:
            self._buffer = h5py.File(self.filename, "a")

    @property
    def _estimated_size(self):
        return (np.prod(np.array(self._dd))) * 8 * len(self.fields) / (1e9)

    @property
    def _estimated_chunk_memory(self):
        return (self.chunksize**3) * 8 / (1e9)

    @property
    def buffer(self):
        """
        Direct access to the underlying HDF5 file at the base of the file structure.
        """
        self._ensure_open()
        return self._buffer

    @property
    def grid(self):
        """
        Direct access to the ``/grid`` root of the HDF5 data. This is where all of the datasets are stored.
        """
        self._ensure_open()

        if self._gridbuffer is None or not self._gridbuffer:
            self._gridbuffer = self._buffer["grid"]
        return self._gridbuffer

    @property
    def is_normalized(self):
        """
        Check if the :py:class:`data_structures.YTHDF5` object is normalized (it's weighted fields are normalized).
        """
        self._ensure_open()

        return self._buffer.attrs["is_normalized"]

    def __bool__(self):
        return self._buffer.__bool__()

    def __str__(self):
        return f"<YTHDF5 File> @ {self.filename}"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        self._ensure_open()
        return self._buffer["grid"][item]

    def __setitem__(self, key, value):
        raise SyntaxError("Values cannot be set by indexing in YTHDF5 objects.")

    def keys(self):
        """
        Alias for ``self.buffer['grid'].keys()``.

        Returns
        -------
        list

        """
        return self.buffer["grid"].keys()

    def _setup(self):
        try:
            self._buffer = h5py.File(self.filename, "a")
        except FileNotFoundError:
            raise IOError(
                f"The parent directories of {self.filename} don't exist. YTHDF5 file was not generated."
            )

        self._gridbuffer = self._buffer.create_group(
            "grid"
        )  # --> create the grid of the dataset.

        for field in self.fields:
            if field in _allowed_fields:
                self._gridbuffer.create_dataset(field, self._dd, dtype="float64")
                self._gridbuffer[field].attrs["unit"] = units[field][0]
            else:
                mylog.warning(
                    f"The field {field} is not recognized. It will be ignored."
                )

        self.buffer.attrs["domain_dimensions"] = self._dd
        self.buffer.attrs["bbox"] = self.bbox
        self.buffer.attrs["chunksize"] = self.chunksize
        self.buffer.attrs["fields"] = self.fields
        self.buffer.attrs[
            "is_normalized"
        ] = False  # tag for renorm / unnorm when needed.
        self.buffer.attrs["model_count"] = 0
        # Generate the chunk id's and the indices of the edges ahead of time to reuse.
        self._buffer.create_group("chunks")
        self._cm = construct_chunks(self._dd, self.chunksize)
        mylog.info(f"Decomposed {self} domain into {self._cm.shape[-1]} chunks.")
        self._buffer["chunks"].create_dataset(
            "chunkmap", self._cm.shape, dtype="uint32"
        )

        self._buffer.close()
        mylog.info(f"Generated YT-HDF5 file at {self.filename}.")

    def survey_memory(self):
        """
        Prints a survey of the expected memory and disk usage of the :py:class:`data_structures.YTHDF5` instance.

        If ``psutil`` is installed, additional information is provided regarding the systems capacity to execute the
        chunked operations.
        """
        mylog.info(f"MEMORY SURVEY: {self.filename}")
        mylog.info(f"Total size: {np.round(self._estimated_size,decimals=4)} GB.")
        mylog.info(
            f"Chunk size: {np.round(self._estimated_chunk_memory,decimals=4)} GB."
        )

        try:
            import psutil

            mylog.info(
                f"Free memory: {np.round(psutil.virtual_memory().available/1e9,decimals=3)} GB"
            )

            if psutil.virtual_memory().available / 1e9 < self._estimated_chunk_memory:
                mylog.warning(
                    "Free memory may be insufficient for chunked operations. Processes at this chunksize may fail."
                )
            else:
                pass
        except ImportError:
            pass  # The user doesn't have psutils installed, we don't want to force them to do so.

    def add_model(self, model, center, velocity, normalize=True, level=0):
        """
        Add the :py:class:`model.ClusterModel` instance to the :py:class:`data_structures.YTHDF5` instance.

        Parameters
        ----------
        model: :py:class:`model.ClusterModel`
            The model to add to the :py:class:`data_structures.YTHDF5` instance.
        center: array-like
            The center of the cluster in the coordinates of :py:attr:`data_structures.YTHDF5.bbox`.
        velocity: array-like
            The COM velocity of the cluster in the coordinates of :py:attr:`data_structures.YTHDF5.bbox`.
        """
        center, velocity = ensure_ytarray(center, "kpc"), ensure_ytarray(
            velocity, "kpc/Myr"
        )
        _model_id = self.buffer.attrs[
            "model_count"
        ]  # --> This is for the mixing weight.

        _relative_bbox = self.bbox - center.d.reshape((3, 1))

        if self.is_normalized:
            mylog.info(
                f"Un-normalizing {self.__str__()} to accommodate new model {model}."
            )
            self._unnormalize(level=level)

        mylog.info(f"Adding {model} to {self.__str__()}")
        mylog.info(
            f"\tPos: {[np.round(j,decimals=2) for j in center.d]} kpc, Vel: {[np.round(j,decimals=2) for j in velocity.to_value('km/s')]} km/s"
        )

        # setup the coordinates to maximize ease of interpolation.
        _rr = model["radius"].to_value("kpc")
        _dens = model["density"].to_value(
            units["density"][0]
        )  # needed for normalization

        # secure the chunkmap, make sure it exists and is in order.
        _chunkmap = self._cm

        for field in tqdm(
            self.fields,
            desc=f"Writing {model} to YTHDF5",
            position=level,
            leave=False,
            disable=(not cgparams["system"]["display"]["progress_bars"]),
        ):
            if field in dens_weighted_fields + additive_fields:
                if field in dens_weighted_fields:
                    _yy = _dens * model[field].to_value(
                        units[field][0], equivalence=units[field][1]
                    )
                else:
                    _yy = model[field].to_value(
                        units[field][0], equivalence=units[field][1]
                    )
                self._add_field(
                    _rr, _yy, field, _relative_bbox, _chunkmap, level=level + 1
                )

        mylog.info(f"Core fields of {model} where written to {self}.")
        self._add_model_velocity_fields(
            model, velocity, _model_id, _chunkmap, _relative_bbox, level=level
        )

        if normalize:
            # --> We are going to normalize the weighted fields.
            self._normalize(level=level)
        else:
            pass

        self.buffer.attrs["model_count"] += 1
        self.buffer.close()

    def add_ICs(self, ics):
        """
        Add an entire :py:class:`ics.ClusterICs` instance to the :py:class:`data_structures.YTHDF5` buffer.

        Parameters
        ----------
        ics: :py:class:`ics.ClusterICs`
            The initial conditions to add to the HDF5 buffer.
        """
        mylog.info(f"Adding {ics.basename} to {self}.")

        if self.is_normalized:
            mylog.info(f"Un-normalizing {self.__str__()} to accommodate new ics {ics}.")
            self._unnormalize(level=1)

        for ic_id, ic_model in enumerate(
            tqdm(
                ics.profiles,
                desc=f"Writing {ics} to YTHDF5",
                position=0,
                leave=False,
                disable=(not cgparams["system"]["display"]["progress_bars"]),
            )
        ):
            model = ClusterModel.from_h5_file(ic_model)
            center, velocity = ics.center[ic_id], ics.velocity[ic_id]

            self.add_model(model, center, velocity, normalize=False, level=1)

        self._normalize()
        self.buffer.close()

    def _add_field(self, r, y, fieldname, bbox, chunkmap, level=1):
        # Interpolate and pass to the Cython level.
        _, _, _, _, _, k, _, n, t, c, _, _, _, _ = dfitpack.fpcurf0(
            r, y, 3, w=None, xb=r[0], xe=r[-1], s=0.0
        )
        _buffer_obj = self.grid[fieldname]
        dump_field_to_hdf5(
            _buffer_obj, bbox, self._dd, chunkmap, t, c, k, level, fieldname
        )

    def _add_model_velocity_fields(self, model, velocity, id, cm, rbb, level=0):
        mylog.info(f"Managing non-fields of {model}")

        _available_nonfields = [
            key
            for key, value in dens_weighted_nonfields.items()
            if value[0] in model.fields
        ]
        _rr = model["radius"].to_value("kpc")
        for non_field in tqdm(
            _available_nonfields,
            desc="Writing non-fields to YTHDF5",
            position=level,
            leave=False,
            disable=(not cgparams["system"]["display"]["progress_bars"]),
        ):
            # we add these if they don't already exist.
            if non_field not in list(self.grid.keys()):
                self._gridbuffer.create_dataset(non_field, self._dd, dtype="float64")
                self._gridbuffer[non_field].attrs["unit"] = units[non_field][0]

            # non-fields are corrected and density weighted.
            field_params = dens_weighted_nonfields[non_field]

            if field_params[2] == "velocity":
                # this is weighted by velocity
                _yy = velocity[field_params[1]] * model[field_params[0]].to_value(
                    units[field_params[0]][0], equivalence=units[field_params[0]][1]
                )
            elif field_params[2] == "model_id":
                _yy = id * model[field_params[0]].to_value(
                    units[field_params[0]][0], equivalence=units[field_params[0]][1]
                )

            self._add_field(_rr, _yy, non_field, rbb, cm, level=level + 1)

    def _normalize(self, level=0):
        mylog.info(f"Renormalizing density weighted fields for {self}")

        for field in tqdm(
            dens_weighted_fields + list(dens_weighted_nonfields.keys()),
            desc=f"Normalizing {self}.",
            position=level,
            leave=False,
            disable=(not cgparams["system"]["display"]["progress_bars"]),
        ):
            renormalize(
                self.grid[field], self.grid["density"], self._cm, level + 1, field
            )

        self.buffer.attrs["is_normalized"] = True

    def _unnormalize(self, level=0):
        mylog.info(f"Un-normalizing density weighted fields for {self}")

        for field in tqdm(
            dens_weighted_fields + list(dens_weighted_nonfields.keys()),
            desc=f"Un-normalizing {self}.",
            position=level,
            leave=False,
            disable=(not cgparams["system"]["display"]["progress_bars"]),
        ):
            unnormalize(
                self.grid[field], self.grid["density"], self._cm, level + 1, field
            )

        self.buffer.attrs["is_normalized"] = False

    def create_dataset(self, simulation_time=0.0, dataset_name="unnamed"):
        """
        Read the :py:class:`data_structures.YTHDF5` instance as a YT dataset.

        Parameters
        ----------
        simulation_time: float, optional
            The simulation time at which to load the dataset. This makes relatively little impact on the results of any analysis;
            however, it may impact redshift dependent measures. Default is ``0``.
        dataset_name: str, optional
            The name of the dataset. By default, this is ``"unnamed"``.

        Returns
        -------
        yt dataset

        """
        mylog.info(f"Passing {self} to yt.")
        import yt

        return yt.load_hdf5_file(
            self.filename,
            "/grid",
            bbox=self.bbox,
            dataset_arguments={
                "length_unit": "kpc",
                "mass_unit": "Msun",
                "time_unit": "Myr",
                "magnetic_unit": "G",
                "velocity_unit": "kpc/Myr",
                "sim_time": simulation_time,
                "dataset_name": dataset_name,
            },
        )
