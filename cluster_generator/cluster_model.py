from collections import OrderedDict
from six import add_metaclass
from yt import savetxt, mylog, YTArray
from yt.funcs import ensure_list
import h5py
import os
import numpy as np

equilibrium_model_registry = {}

class RegisteredClusterModel(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "_type_name"):
            equilibrium_model_registry[cls._type_name] = cls

@add_metaclass(RegisteredClusterModel)
class ClusterModel(object):

    def __init__(self, num_elements, fields):
        self.num_elements = num_elements
        self.fields = fields

    @classmethod
    def from_h5_file(cls, filename):
        r"""
        Generate an equilibrium model from an HDF5 file. 

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.

        Examples
        --------
        >>> from cluster_generator import ClusterModel
        >>> hse_model = ClusterModel.from_h5_file("hse_model.h5")
        """
        f = h5py.File(filename)

        model_type = f["model_type"].value
        num_elements = f["num_elements"].value
        fnames = list(f['fields'].keys())

        f.close()

        fields = OrderedDict()
        for field in fnames:
            fields[field] = YTArray.from_hdf5(filename, dataset_name=field,
                                              group_name="fields").in_base("galactic")

        return equilibrium_model_registry[model_type](num_elements, fields)

    def __getitem__(self, key):
        return self.fields[key]

    def keys(self):
        return self.fields.keys()

    def write_model_to_ascii(self, output_filename, in_cgs=False, overwrite=False):
        """
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % output_filename)
        field_list = list(self.fields.keys())
        num_fields = len(field_list)
        name_fmt_str = " Fields\n"+" %s\t"*(num_fields-1)+"%s"
        header = name_fmt_str % tuple(field_list)

        if in_cgs:
            fields = OrderedDict()
            for k, v in self.fields.items():
                fields[k] = v.in_cgs()
        else:
            fields = self.fields

        savetxt(output_filename, list(fields.values()), header=header)

    def write_model_to_h5(self, output_filename, in_cgs=False, overwrite=False):
        """
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % output_filename)
        f = h5py.File(output_filename, "w")
        f.create_dataset("model_type", data=self._type_name)
        f.create_dataset("num_elements", data=self.num_elements)
        f.close()
        for field in list(self.fields.keys()):
            if in_cgs:
                if field == "temperature":
                    fd = self.fields[field].to_equivalent("K", "thermal")
                else:
                    fd = self.fields[field]
                fd.in_cgs().write_hdf5(output_filename, dataset_name=field, 
                                       group_name="fields")
            else:
                self.fields[field].write_hdf5(output_filename, dataset_name=field,
                                              group_name="fields")

    def set_field(self, name, value):
        """
        Set a field with name *name* to value *value*, which is a YTArray.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, YTArray):
            raise TypeError("value needs to be a YTArray")
        if value.size == self.num_elements:
            if name in self.fields:
                mylog.warning("Overwriting field %s." % name)
            self.fields[name] = value
        else:
            raise ValueError("The length of the array needs to be %d elements!"
                             % self.num_elements)

gadget_dm_fields = ["Coordinates", "Velocities", "Masses"]
gadget_gas_fields = ["Coordinates", "Velocities", "Masses",
                     "InternalEnergy", "MagneticField"]

gadget_field_map = {"Coordinates": "particle_position",
                    "Velocities": "particle_velocity",
                    "Masses": "particle_mass",
                    "InternalEnergy": "particle_thermal_energy",
                    "MagneticField": "particle_magnetic_field"}

gadget_field_units = {"Coordinates": "kpc",
                      "Velocities": "km/s",
                      "Masses": "1e10*Msun",
                      "InternalEnergy": "km**2/s**2",
                      "MagneticField": "gauss"}

class ClusterParticles(object):
    def __init__(self, particle_types, fields):
        self.particle_types = ensure_list(particle_types)
        self.num_particles = {}
        for ptype in self.particle_types:
            self.num_particles[ptype] = fields[ptype, "particle_mass"].size
        self.fields = fields

    @classmethod
    def from_h5_file(cls, filename):
        r"""
        Generate cluster particles from an HDF5 file.

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.

        Examples
        --------
        >>> from cluster_generator import ClusterParticles
        >>> dm_particles = ClusterParticles.from_h5_file("dm_particles.h5")
        """
        names = {}
        f = h5py.File(filename)
        particle_types = list(f.keys())
        for ptype in f:
            names[ptype] = list(f[ptype].keys())
        f.close()

        fields = OrderedDict()
        for ptype in particle_types:
            for field in names[ptype]:
                fields[ptype, field] = YTArray.from_hdf5(filename, dataset_name=field,
                                                         group_name=ptype).in_base("galactic")
        return cls(particle_types, fields)

    @classmethod
    def from_gadget_ics(cls, filename):
        fields = OrderedDict()
        f = h5py.File(filename, "r")
        particle_types = []
        if "PartType0" in f:
            particle_types.append("gas")
            gas = f["PartType0"]
            for field in gadget_gas_fields:
                if field in gas:
                    fd = gadget_field_map[field]
                    units = gadget_field_units[field]
                    fields["gas", fd] = YTArray(gas[field], units).in_base("galactic")
        if "PartType1" in f:
            particle_types.append("dm")
            dm = f["PartType1"]
            for field in gadget_dm_fields:
                if field in dm:
                    fd = gadget_field_map[field]
                    units = gadget_field_units[field]
                    fields["dm", fd] = YTArray(dm[field], units).in_base("galactic")
        f.close()
        return cls(particle_types, fields)

    def write_particles_to_h5(self, output_filename, in_cgs=False, overwrite=False):
        """
        Write the particles to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the particles to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % output_filename)
        if overwrite and os.path.exists(output_filename):
            os.remove(output_filename)
        for field in self.fields:
            if in_cgs:
                fd = self.fields[field].in_cgs()
            else:
                fd = self.fields[field]
            fd.write_hdf5(output_filename, dataset_name=field[1],
                          group_name=field[0])

    def __add__(self, other):
        fields = self.fields.copy()
        fields.update(other.fields)
        particle_types = self.particle_types + other.particle_types
        return ClusterParticles(particle_types, fields)

    def add_offsets(self, r_ctr, v_ctr):
        if not isinstance(r_ctr, YTArray):
            r_ctr = YTArray(r_ctr, "kpc")
        if not isinstance(v_ctr, YTArray):
            v_ctr = YTArray(v_ctr, "kpc/Myr")
        for ptype in self.particle_types:
            self.fields[ptype, "particle_position"] += r_ctr
            self.fields[ptype, "particle_velocity"] += v_ctr

    def _clip_to_box(self, ptype, box_size):
        pos = self.fields[ptype, "particle_position"]
        return ~np.logical_or((pos < 0.0).any(axis=1), (pos > box_size).any(axis=1))

    def _write_gadget_fields(self, ptype, h5_group, idxs):
        for field in gadget_gas_fields:
            my_field = gadget_field_map[field]
            if (ptype, my_field) in self.fields:
                units = gadget_field_units[field]
                data = self.fields[ptype, my_field][idxs].in_units(units).d.astype("float32")
                h5_group.create_dataset(field, data=data)

    def write_to_gadget_ics(self, ic_filename, box_size, overwrite=False):
        if os.path.exists(ic_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % ic_filename)
        num_particles = 0
        num_gas_particles = 0
        num_dm_particles = 0
        f = h5py.File(ic_filename, "w")
        if "gas" in self.particle_types:
            gidxs = self._clip_to_box("gas", box_size)
            num_gas_particles = gidxs.sum()
            gasg = f.create_group("PartType0")
            self._write_gadget_fields("gas", gasg, gidxs)
            ids = np.arange(num_gas_particles)+num_particles
            gasg.create_dataset("ParticleIDs", data=ids.astype('uint32'))
            num_particles += num_gas_particles
        if "dm" in self.particle_types:
            didxs = self._clip_to_box("dm", box_size)
            num_dm_particles = didxs.sum()
            dmg = f.create_group("PartType1")
            self._write_gadget_fields("dm", dmg, didxs)
            ids = np.arange(num_dm_particles)+num_particles
            dmg.create_dataset("ParticleIDs", data=ids.astype('uint32'))
            num_particles += num_dm_particles
        f.flush()
        hg = f.create_group("Header")
        hg.attrs["Time"] = 0.0
        hg.attrs["Redshift"] = 0.0
        hg.attrs["BoxSize"] = box_size
        hg.attrs["Omega0"] = 0.0
        hg.attrs["OmegaLambda"] = 0.0
        hg.attrs["HubbleParam"] = 1.0
        hg.attrs["NumPart_ThisFile"] = np.array([num_gas_particles, num_dm_particles], 
                                                dtype='uint32')
        hg.attrs["NumPart_Total"] = hg.attrs["NumPart_ThisFile"]
        hg.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype='uint32')
        hg.attrs["NumFilesPerSnapshot"] = 1
        hg.attrs["MassTable"] = np.zeros(6)
        hg.attrs["Flag_Sfr"] = 0
        hg.attrs["Flag_Cooling"] = 0
        hg.attrs["Flag_StellarAge"] = 0
        hg.attrs["Flag_Metals"] = 0
        hg.attrs["Flag_Feedback"] = 0
        hg.attrs["Flag_DoublePrecision"] = 0
        hg.attrs["Flag_IC_Info"] = 0
        f.flush()
        f.close()

    def set_field(self, ptype, name, value):
        """
        Set a field with name *name* to value *value*, which is a YTArray.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, YTArray):
            raise TypeError("value needs to be a YTArray")
        num_particles = self.num_particles[ptype]
        if value.size == num_particles:
            if (ptype, name) in self.fields:
                mylog.warning("Overwriting field (%s, %s)." % (ptype, name))
            self.fields[ptype, name] = value
        else:
            raise ValueError("The length of the array needs to be %d particles!"
                             % num_particles)

    def __getitem__(self, key):
        return self.fields[key]

    def keys(self):
        return self.fields.keys()
