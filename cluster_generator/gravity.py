"""
Methods for computations regarding gravity.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from unyt import unyt_array

from cluster_generator.utils import \
    integrate, mylog, G, integrate_toinf
import os
import h5py
from collections import OrderedDict

# -------------------------------------------------------------------------------------------------------------------- #
#  Default attributes================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
_default_interpolation_function = lambda x: x/(1+x)
_default_a_0 = unyt_array(1.2e-10,"m/s**2")
# -------------------------------------------------------------------------------------------------------------------- #
# Functions ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
class Potential:
    """
    The ``Potential`` class is used as a buffer between ``Model`` objects and ``Profile`` objects. The ``Potential`` houses
    the various methodologies for computing the correct gravitational potential. This includes the ability to compute MOND potentials.

    Attributes
    ----------
    fields: dict
        The fields specified to the ``Potential`` object.
    gravity: str
        The type of gravity that is in use. Options are ``AQUAL``, ``QUMOND``, or ``Newtonian``.
    self.num_elements: int
        The number of points at which the profiles were sampled to generate the fields.

    """
    _keep_units = ["entropy", "electron_number_density",
                   "magnetic_field_strength"]

    def __init__(self, fields, gravity, attrs):
        # - Initializing base attributes - #
        self.fields = fields
        self.gravity = gravity

        # - Derived attributes -#
        self.num_elements = len(self.fields["radius"])

        if "potential" not in self.fields:
            self.fields["potential"] = None

        self.attrs = attrs

        if gravity != "Newtonian":
            if "interp_function" not in self.attrs:
                mylog.warning(f"Gravity {self.gravity} requires kwarg `interp_function`. Setting to default...")
                self.attrs["interp_function"] = _default_interpolation_function
            if "a_0" not in self.attrs:
                mylog.warning(f"Gravity {self.gravity} requires kwarg `a_0`. Setting to default...")
                self.attrs["a_0"] = _default_a_0

    def __repr__(self):
        return f"Potential object; gravity={self.gravity}"

    def __str__(self):
        return f"Potential object; gravity={self.gravity}, fields={list(self.fields.keys())}"

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, item):
        return self.fields[item]

    def __setitem__(self, key, value):
        self.fields[key] = value

    # ---------------------------------------------------------------------------------------------------------------- #
    # Properties ===================================================================================================== #
    # ---------------------------------------------------------------------------------------------------------------- #
    @property
    def pot(self):
        """The potential array of the ``Potential``."""
        if self.fields["potential"] is not None:
            return self.fields["potential"]
        else:
            try:
                self.potential()
                return self.fields["potential"]
            except ValueError:
                raise ValueError("Failed to compute a potential from available fields.")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Class Methods ================================================================================================== #
    # ---------------------------------------------------------------------------------------------------------------- # 
    @classmethod
    def from_fields(cls, fields, gravity="Newtonian", **kwargs):
        """
        Initializes a ``Potential`` object from raw input fields ``fields``.

        Parameters
        ----------
        fields: dict
            The fields of data from which to generate the ``Potential`` object.
        gravity: str, optional, default="Newtonian"
            The type of gravity to use in computations. Options are ``Newtonian``,``AQUAL``, or ``QUMOND``.
    
        Returns
        -------
        Potential
            The resultant potential.
        """
        _methods = {
            "mdr": ["radius", "total_mass", "total_density"]
        }
        #  Logging
        # ----------------------------------------------------------------------------------------------------------------- #
        mylog.info(f"Computing model potential in {gravity} gravity.")

        #  Sanity Check
        # ----------------------------------------------------------------------------------------------------------------- #
        if all(any(field not in fields for field in req_fields) for req_fields in list(_methods.values())):
            raise ValueError(f"The fields {list(fields.keys())} are not sufficient for a potential computation.")
        else:
            _used_method = [key for key, value in _methods.items() if all(req_field in fields for req_field in value)][
                0]
            mylog.info(f"Computation of potential is using {_used_method} for computation.")

        #  Computing Potential
        # ----------------------------------------------------------------------------------------------------------------- #
        obj = cls(fields, gravity,kwargs)  # initialize the object
        getattr(obj, f"_find_from_{_used_method}")()  # pass the computation off to subfunctions

        return obj

    @classmethod
    def from_h5_file(cls, filename):
        r"""
        Generates the ``Potential`` object from an ``HDF5`` file.

        Parameters
        ----------
        filename : string
            The name of the file to read the ``Potential`` from.
        """
        # - Grabbing base data -#
        with h5py.File(filename, "r") as f:
            fnames = list(f['fields'].keys())
            gravity = f.attrs["gravity"]

        fields = OrderedDict()
        mylog.info(f"Found {len(fnames)} in {filename}: {fnames}")
        for field in fnames:  # -> converting fields to unyt_arrays.
            a = unyt_array.from_hdf5(filename, dataset_name=field,
                                     group_name="fields")
            fields[field] = unyt_array(a.d, str(a.units))
            if field not in cls._keep_units:  # --> using data conversion.
                fields[field].convert_to_base("galactic")

        # - Determining rmin / rmax and masking. -#
        r_min = 0.0
        r_max = fields["radius"][-1].d * 2
        mask = np.logical_and(fields["radius"].d >= r_min,
                              fields["radius"].d <= r_max)
        for field in fnames:
            fields[field] = fields[field][mask]

        model = cls(fields, gravity,{})
        return model

    # ---------------------------------------------------------------------------------------------------------------- #
    # Methods ======================================================================================================== #
    # ---------------------------------------------------------------------------------------------------------------- # 

    def potential(self):
        """
        Computes the potential from available fields.


        Returns
        -------
        Potential
            The resultant potential.
        """
        _methods = {
            "mdr": ["radius", "total_mass", "total_density"]
        }
        #  Logging
        # ----------------------------------------------------------------------------------------------------------------- #
        mylog.info(f"Computing model potential in {self.gravity} gravity.")

        #  Sanity Check
        # ----------------------------------------------------------------------------------------------------------------- #
        if all(any(field not in self.fields for field in req_fields) for req_fields in list(_methods.values())):
            raise ValueError(
                f"The self.fields {list(self.fields.keys())} are not sufficient for a potential computation.")
        else:
            _used_method = \
            [key for key, value in _methods.items() if all(req_field in self.fields for req_field in value)][
                0]
            mylog.info(f"Computation of potential is using {_used_method} for computation.")

        #  Computing Potential
        # ----------------------------------------------------------------------------------------------------------------- #
        getattr(self, f"_find_from_{_used_method}")()  # pass the computation off to subfunctions
        return None  # Sentinel

    def _find_from_mdr(self):
        """ Compute the potential from MDR."""

        if self.gravity == "Newtonian":
            # -------------------------------------------------------------------------------------------------------- #
            #  Newtonian Implementation                                                                                #
            # -------------------------------------------------------------------------------------------------------- #
            # - Pulling arrays 
            rr = self.fields["radius"].d

            mylog.info(f"Integrating gravitational potential profile. gravity={self.gravity}.")
            tdens_func = InterpolatedUnivariateSpline(rr, self.fields["total_density"].d)

            # - Computing - #
            gpot_profile = lambda r: tdens_func(r) * r

            gpot1 = self.fields["total_mass"] / self.fields["radius"]
            gpot2 = unyt_array(4. * np.pi * integrate(gpot_profile, rr), "Msun/kpc")

            # - Finishing computation - #

            self.fields["potential"] = -G * (gpot1 + gpot2)
            self.fields["potential"].convert_to_units("kpc**2/Myr**2")

        elif self.gravity == "AQUAL":
            # -------------------------------------------------------------------------------------------------------- #
            #  AQUAL Implementation                                                                                    #
            # -------------------------------------------------------------------------------------------------------- #
            mylog.info(f"Integrating gravitational potential profile. gravity={self.gravity}.")
            # - pulling arrays
            rr = self["radius"].d
            tmass = self["total_mass"].d
            self.attrs["a_0"].to("kpc/Myr**2")

            # - Building the gamma function - #
            gamma_func = InterpolatedUnivariateSpline(rr,G*tmass/(float(self.attrs["a_0"].d)*(rr**2)))
            self["gamma"] = unyt_array(gamma_func(rr))

            # - generating guess solution - #
            mylog.info(f"Creating AQUAL guess solution for implicit equation...")

            Gamma_func = lambda x: (1/2)*(gamma_func(x)+np.sqrt(gamma_func(x)**2 + 4*gamma_func(x))) #-> big gamma del Phi / a_0
            _guess = Gamma_func(rr)


            # - solving - #
            mylog.info(f"Optimizing implicit solution...")
            _fsolve_function = lambda x: x*self.attrs["interp_function"](x) - self["gamma"]

            _Gamma_solution = fsolve(_fsolve_function,x0=_guess)

            Gamma = InterpolatedUnivariateSpline(rr,_Gamma_solution)

            # - Performing the integration process - #
            gpot2 = self.attrs["a_0"].d*unyt_array(integrate_toinf(Gamma, rr), "(kpc**2)/Myr**2")

            # - Finishing computation - #
            self.fields["potential"] = -gpot2
            self.fields["potential"].convert_to_units("kpc**2/Myr**2")

            self["guess_potential"] = -self.attrs["a_0"].d*unyt_array(integrate_toinf(Gamma_func, rr), "(kpc**2)/Myr**2")
            self.fields["potential"].convert_to_units("kpc**2/Myr**2")
        elif self.gravity == "QUMOND":
            # -------------------------------------------------------------------------------------------------------- #
            #  QUMOND Implementation                                                                                   #
            # -------------------------------------------------------------------------------------------------------- #
            # - Pulling arrays
            rr = self["radius"].d
            tmass = self["total_mass"].d
            ## -- Preparing for Execution -- ##
            mylog.info(f"Integrating gravitational potential profile. gravity={self.gravity}.")
            gamma_func = InterpolatedUnivariateSpline(rr,G*tmass/(float(self.attrs["a_0"].d)*(rr**2)))
            self["gamma"] = unyt_array(gamma_func(rr))
            gpot_profile = lambda r: - gamma_func(r) * float(self.attrs["a_0"].d) *self.attrs["interp_function"](gamma_func(r))

            # - Performing the integration process - #
            gpot2 = unyt_array(integrate_toinf(gpot_profile, rr), "(kpc**2)/Myr**2")

            # - Finishing computation - #
            self.fields["potential"] = gpot2
            self.fields["potential"].convert_to_units("kpc**2/Myr**2")
        else:
            raise ValueError(f"The gravity type {self.gravity} is not a valid gravity type.")

    def plot(self, rmin=None, rmax=None, fig=None, ax=None, **kwargs):
        """
        Plot a field vs radius from this model using Matplotlib.

        Parameters
        ----------
        field : string
            The field to plot.
        r_min : float
            The minimum radius of the plot in kpc.
        r_max : float
            The maximum radius of the plot in kpc.
        fig : Matplotlib Figure
            The figure to plot in. Default; None, in which case
            one will be generated.
        ax : Matplotlib Axes
            The axes to plot in. Default: None, in which case
            one will be generated.

        Returns
        -------
        The Figure, Axes tuple used for the plot.
        """
        import matplotlib.pyplot as plt
        plt.rc("font", size=18)
        plt.rc("axes", linewidth=2)
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        if ax is None:
            ax = fig.add_subplot(111)
        ax.loglog(self["radius"], -self.pot, **kwargs)
        ax.set_xlim([rmin, rmax])
        ax.set_xlabel("Radius (kpc)")
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        return fig, ax

    def write_potential_to_h5(self, output_filename, in_cgs=False, overwrite=False):
        r"""
        Write the ``Potential`` object to an ``h5`` file.

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
        f.create_dataset("num_elements", data=self.num_elements)
        f.attrs["unit_system"] = "cgs" if in_cgs else "galactic"
        f.attrs["gravity"] = self.gravity
        f.close()
        r_min = 0.0
        r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(self.fields["radius"].d >= r_min,
                              self.fields["radius"].d <= r_max)

        for k, v in self.fields.items():
            if in_cgs:
                if k == "temperature":
                    fd = v[mask].to_equivalent("K", "thermal")
                elif k not in self._keep_units:
                    fd = v[mask].in_cgs()
                else:
                    fd = v[mask]
            else:
                fd = v[mask]
            fd.write_hdf5(output_filename, dataset_name=k,
                          group_name="fields")

        if self.fields["potential"] is not None:
            fd = self.fields["potential"]
            fd.write_hdf5(output_filename, dataset_name="potential", group_name="fields")
