from collections import OrderedDict
from unyt import unyt_array
import h5py
import os
import numpy as np
from .utils import mylog

equilibrium_model_registry = {}


class RegisteredClusterModel(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "_type_name"):
            equilibrium_model_registry[cls._type_name] = cls


class ClusterModel(metaclass=RegisteredClusterModel):

    def __init__(self, num_elements, fields, parameters=None):
        if parameters is None:
            parameters = {}
        self.num_elements = num_elements
        self.fields = fields
        self.parameters = parameters

    @classmethod
    def from_arrays(cls, model_type, fields, parameters=None):
        return equilibrium_model_registry[model_type](
            fields["radius"].size, fields, parameters=parameters)

    @classmethod
    def from_h5_file(cls, filename, r_min=None, r_max=None):
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
        with h5py.File(filename, "r") as f:
            model_type = f["model_type"][()].decode()
            fnames = list(f['fields'].keys())

            parameters = {}
            if "parameters" in f:
                for k in f["parameters"].keys():
                    parameters[k] = f["parameters"][k][()]

        fields = OrderedDict()
        for field in fnames:
            a = unyt_array.from_hdf5(filename, dataset_name=field,
                                  group_name="fields")
            fields[field] = unyt_array(a.d, str(a.units)).in_base("galactic")
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = fields["radius"][-1].d*2
        mask = np.logical_and(fields["radius"].d >= r_min, 
                              fields["radius"].d <= r_max)
        for field in fnames:
            fields[field] = fields[field][mask]
        num_elements = mask.sum()

        return equilibrium_model_registry[model_type](num_elements, fields,
                                                      parameters=parameters)

    def __getitem__(self, key):
        return self.fields[key]

    def __contains__(self, key):
        return key in self.fields

    def keys(self):
        return self.fields.keys()

    def write_model_to_ascii(self, output_filename, in_cgs=False,
                             overwrite=False):
        r"""
        Write the equilibrium model to an ascii text file. Uses
        AstroPy's QTable to write the file, so that units are
        included.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default: False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default: False.
        """
        from astropy.table import QTable
        fields = {}
        for k, v in self.fields.items():
            if in_cgs:
                fields[k] = v.in_cgs().to_astropy()
            else:
                fields[k] = v.to_astropy()
        t = QTable(fields)
        t.meta['comments'] = f"unit_system={'cgs' if in_cgs else 'galactic'}"
        t.write(output_filename, overwrite=overwrite)

    def write_model_to_h5(self, output_filename, in_cgs=False, overwrite=False):
        r"""
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
            raise IOError(f"Cannot create {output_filename}. It exists and "
                          f"overwrite=False.")
        f = h5py.File(output_filename, "w")
        f.create_dataset("model_type", data=self._type_name)
        f.create_dataset("num_elements", data=self.num_elements)
        f.attrs["unit_system"] = "cgs" if in_cgs else "galactic"
        g = f.create_group("parameters")
        for k, v in self.parameters.items():
            g.create_dataset(k, data=v)
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
                self.fields[field].write_hdf5(output_filename, 
                    dataset_name=field, group_name="fields")

    def set_field(self, name, value):
        r"""
        Set a field with name *name* to value *value*, which is an unyt_array.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, unyt_array):
            raise TypeError("value needs to be an unyt_array")
        if value.size == self.num_elements:
            if name in self.fields:
                mylog.warning("Overwriting field %s." % name)
            self.fields[name] = value
        else:
            raise ValueError("The length of the array needs to be %d elements!"
                             % self.num_elements)
