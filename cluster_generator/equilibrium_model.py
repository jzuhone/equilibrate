from cluster_generator.utils import mylog, \
    YTArray
from collections import OrderedDict
from six import add_metaclass
import h5py
import os

equilibrium_model_registry = {}

class RegisteredEquilibriumModel(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "_type_name") and not cls._skip_add:
            equilibrium_model_registry[cls._type_name] = cls

@add_metaclass(RegisteredEquilibriumModel)
class EquilibriumModel(object):
    _skip_add = False

    def __init__(self, num_elements, fields, geometry):
        self.num_elements = num_elements
        self.fields = fields
        self.geometry = geometry

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
        >>> from cluster_generator import EquilibriumModel
        >>> hse_model = EquilibriumModel.from_h5_file("hse_model.h5")
        """
        f = h5py.File(filename)
        model_type = f["model_type"].value

        geometry = f["geometry"].value

        f.close()

        fields = OrderedDict
        for field in f["fields"]:
            fields[field] = YTArray.from_hdf5(filename, dataset_name=field, group_name="fields")

        num_elements = f["num_elements"].value

        return equilibrium_model_registry[model_type](num_elements, fields, geometry)

    def __getitem__(self, key):
        return self.fields[key]

    def keys(self):
        return self.fields.keys()

    def write_model_to_ascii(self, output_filename):
        """
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        """
        field_list = self.fields.keys()
        num_fields = len(field_list)

        format_str = "%g\t"*(num_fields-1)+"%g\n"
        name_fmt_str = "# " + "%s\t"*(num_fields-1)+"%s\n"
        unit_fmt_str = "# " + "%s\t"*(num_fields-1)+"%s\n"

        outlines = [name_fmt_str % tuple(field_list)]
        units = [str(self[field].units) for field in field_list]
        outlines.append(unit_fmt_str % tuple(units))

        for i in xrange(self.num_elements):
            current_line = [self[field][i] for field in field_list]
            outlines.append(format_str % tuple(current_line))

        f = open(output_filename, "w")
        f.writelines(outlines)
        f.close()

    def write_model_to_h5(self, output_filename, in_cgs=False, clobber=False):
        """
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        clobber : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not clobber:
            raise IOError("Cannot create %s. It exists and clobber=False." % output_filename)
        f = h5py.File(output_filename, "w")
        f.create_dataset("model_type", data=self._type_name)
        f.create_dataset("num_elements", data=self.num_elements)
        f.create_dataset("geometry", data=self.geometry)
        f.close()
        for field in list(self.fields.keys()):
            if in_cgs:
                self.fields[field].in_cgs().write_hdf5(output_filename, dataset_name=field, group_name="fields")
            else:
                self.fields[field].write_hdf5(output_filename, dataset_name=field, group_name="fields")

    def set_field(self, name, value):
        """
        Set a field with name *name* to value *value*, which is a YTArray.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, YTArray):
            raise TypeError("value needs to be a YTArray")
        if len(value) == self.num_elements:
            if name in self.fields:
                mylog.warning("Overwriting field %s." % name)
            self.fields[name] = value
        else:
            raise ValueError("The length of the array needs to be %d elements!"
                             % self.num_elements)


