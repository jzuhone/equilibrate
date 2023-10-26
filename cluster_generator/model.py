"""
CGP Core module containing classes and method related to the :py:class:`model.ClusterModel` class for representing
physically self-consistent galaxy cluster models.
"""
import os
from collections import OrderedDict

import h5py
import numpy as np
import yaml
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array, unyt_quantity

from cluster_generator.particles import ClusterParticles
from cluster_generator.utils import (
    G,
    _enforce_style,
    ensure_ytquantity,
    generate_particle_radii,
    integrate,
    integrate_mass,
    kpc_to_cm,
    mp,
    mu,
    mue,
    mylog,
)
from cluster_generator.virial import VirialEquilibrium

tt = 2.0 / 3.0
mtt = -tt
ft = 5.0 / 3.0
tf = 3.0 / 5.0
mtf = -tf
gamma = ft
et = 8.0 / 3.0
te = 3.0 / 8.0


class ClusterModel:
    r"""
    The :py:class:`model.ClusterModel` class is a comprehensive representation of the cluster being modeled and can be used to generate
    accurate initial conditions. The class is predicated on a fixed number of sample radii in the cluster.

    Parameters
    ----------
    fields: dict of str: :py:class:`unyt.array.unyt_array`
        The fields to attribute to the :py:class:`model.ClusterModel`.
    properties: dict
        Additional properties to pass into the :py:class:`model.ClusterModel` instance. These are found in the :py:attr:`model.ClusterModel.properties` attribute. See
        the other parameters section for details on the available properties.
    dm_virial: VirialEquilibrium, optional
        The virialization class to use for dark matter virialization. By default, it is None and will be set when necessary.
    star_virial: VirialEquilibrium, optional
        The virialization class to use for stellar matter virialization. By default, it is None and will be set when necessary.

    Notes
    -----
    - ``__getitem__`` and ``__contains__`` are aliased down to ``self.fields``. There is no ``__setitem__``, so index
      / key assignment cannot be done. use ``ClusterModel.set_field()`` instead.

    """

    #: The default included fields that can be accessed.
    default_fields = [
        "density",
        "temperature",
        "pressure",
        "total_density",
        "gravitational_potential",
        "gravitational_field",
        "total_mass",
        "gas_mass",
        "dark_matter_mass",
        "dark_matter_density",
        "stellar_density",
        "stellar_mass",
    ]

    _keep_units = ["entropy", "electron_number_density", "magnetic_field_strength"]

    def __init__(self, fields, properties=None, dm_virial=None, star_virial=None):
        """Initialize the :py:class:`model.ClusterModel` instance."""
        #: The ``fields`` associated with the ``Potential`` object.
        self.fields = fields
        #: The number of elements in each ``field`` array.
        self.num_elements = len(self.fields["radius"])
        if not properties:
            self._properties = {}
        else:
            self._properties = properties

        if "meth" not in self._properties:
            self._properties["meth"] = {}

        self._dm_virial = dm_virial
        self._star_virial = star_virial

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, item):
        return self.fields[item]

    def __setitem__(self, key, value):
        self.fields[key] = value

    def keys(self):
        """Equivalent to ``self.fields.keys()``"""
        return self.fields.keys()

    def items(self):
        """Equivalent to ``self.fields.items()``"""
        return self.fields.items()

    def values(self):
        """Equivalent to ``self.fields.values()``"""
        return self.fields.values()

    @property
    def dm_virial(self):
        """
        The :py:class:`virial.VirialEquilibrium` instance associated with the dark matter virialization process.
        """
        if self._dm_virial is None:
            self._dm_virial = VirialEquilibrium(self, "dark_matter")
        return self._dm_virial

    @property
    def star_virial(self):
        """
        The :py:class:`virial.VirialEquilibrium` instance associated with the stellar virialization process.
        """
        if self._star_virial is None and "stellar_density" in self:
            self._star_virial = VirialEquilibrium(self, "stellar")
        return self._star_virial

    @property
    def properties(self):
        """
        The properties of the :py:class:`model.ClusterModel` instance. When passed to the :py:class:`model.ClusterModel` object,
        the structure should be as a dictionary with the keys listed below and structure described therein. All of these are optional.

        Parameters
        ----------
        gravity: dict or str
            If a string is passed, it should be a gravity name corresponding to the desired gravity theory. All other properties
            of the gravitational theory will be taken as the standard default. If the value is another dictionary, the name of
            the preferred gravitational theory should be associated with the key ``'name'``. Further parameters should be supplied
            as they are described in the documentation for the corresponding gravitational theory.

        Notes
        -----
        In addition to the user specified properties, additional properties may be found in the ``'meth'`` key of the attribute.
        These are specified behind the scenes and dictate the behavior of certain backend behaviors. It is ill-advised to alter
        these properties without a thorough understanding of what they do.

        """
        return self._properties

    @classmethod
    def from_arrays(cls, fields, stellar_density=None, **kwargs):
        r"""
        Initialize the :py:class:`model.ClusterModel` from ``fields`` alone.

        Parameters
        ----------
        fields: dict of str: :py:class:`unyt.array.unyt_array`
            The fields from which to generate the model.

            .. admonition:: development note

                The ``fields`` parameter must be self-consistent in its definition, and must contain a ``radius`` key, from
                which the class assesses the number of elements.
        stellar_density: radial_profiles.RadialProfile, optional
            The stellar density component.
        kwargs:
            Additional attributes to pass through to the :py:attr:`model.ClusterModel.properties` attribute.

        Returns
        -------
        ClusterModel


        See Also
        --------
        :py:meth:`~model.ClusterModel.from_dens_and_tden`,:py:meth:`~model.ClusterModel.from_dens_and_temp`,:py:meth:`~model.ClusterModel.from_dens_and_entr`,:py:meth:`~model.ClusterModel.from_h5_file`

        """
        kwargs = _force_method(kwargs, "from_arrays")
        kwargs["meth"]["profiles"] = {}

        if stellar_density is not None:
            kwargs["meth"]["profiles"]["stellar_density"] = stellar_density
            fields["stellar_density"] = stellar_density(fields["radius"].d)
        else:
            pass

        return cls(fields, properties=kwargs)

    @classmethod
    def from_h5_file(cls, filename, r_min=None, r_max=None):
        r"""
        Initialize a :py:class:`model.ClusterModel` instance from HDF5 file.

        Parameters
        ----------
        filename : str
            The name of the file to read the model from.
        r_min: float, optional
            The minimum radius
        r_max: float, optional
            The maximal radius

        Notes
        -----
        .. attention::

            The :py:meth:`model.ClusterModel.from_h5_file` method will seek both ``filename`` and ``filename.properties``. If
            the later cannot be found, the :py:class:`model.ClusterModel` will lose its attributes and issue a warning.

        Examples
        --------
        >>> from cluster_generator import ClusterModel
        >>> hse_model = ClusterModel.from_h5_file("hse_model.h5")

        """
        import dill as pickle

        from cluster_generator.virial import VirialEquilibrium

        with h5py.File(filename, "r") as f:
            fnames = list(f["fields"].keys())
            get_dm_virial = "dm_df" in f
            get_star_virial = "star_df" in f

        properties_path = f"{filename}.properties"
        try:
            with open(properties_path, "rb") as fpkl:
                attrs = pickle.load(fpkl)

        except FileNotFoundError:
            mylog.warning(f"Failed to load properties file {properties_path}.")
            attrs = {}
        except SystemError:
            mylog.warning("Loading pickled file from unmatched python version.")
            attrs = {}

        fields = OrderedDict()
        for field in fnames:
            a = unyt_array.from_hdf5(filename, dataset_name=field, group_name="fields")
            fields[field] = unyt_array(a.d, str(a.units))
            if field not in cls._keep_units:
                fields[field].convert_to_base("galactic")

        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = fields["radius"][-1].d * 2
        mask = np.logical_and(fields["radius"].d >= r_min, fields["radius"].d <= r_max)
        for field in fnames:
            fields[field] = fields[field][mask]

        model = cls(fields, properties=attrs)

        if get_dm_virial:
            mask = np.logical_and(
                fields["radius"].d >= r_min, fields["radius"].d <= r_max
            )
            df = unyt_array.from_hdf5(filename, dataset_name="dm_df")[mask]
            model._dm_virial = VirialEquilibrium(model, ptype="dark_matter", df=df)

        if get_star_virial:
            mask = np.logical_and(
                fields["radius"].d >= r_min, fields["radius"].d <= r_max
            )
            df = unyt_array.from_hdf5(filename, dataset_name="star_df")[mask]
            model._star_virial = VirialEquilibrium(model, ptype="stellar", df=df)

        return model

    @classmethod
    def _from_scratch(cls, fields, stellar_density=None, **kwargs):
        """Load the cluster instance from total density, total mass, and radius."""
        rr = fields["radius"].d
        mylog.info("Integrating gravitational potential profile.")
        tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)
        gpot_profile = lambda r: tdens_func(r) * r
        gpot1 = fields["total_mass"] / fields["radius"]
        gpot2 = unyt_array(4.0 * np.pi * integrate(gpot_profile, rr), "Msun/kpc")
        fields["gravitational_potential"] = -G * (gpot1 + gpot2)
        fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

        if "density" in fields and "gas_mass" not in fields:
            mylog.info("Integrating gas mass profile.")
            m0 = fields["density"].d[0] * rr[0] ** 3 / 3.0
            fields["gas_mass"] = unyt_array(
                4.0 * np.pi * cumtrapz(fields["density"] * rr * rr, x=rr, initial=0.0)
                + m0,
                "Msun",
            )

        if stellar_density is not None:
            fields["stellar_density"] = unyt_array(stellar_density(rr), "Msun/kpc**3")
            mylog.info("Integrating stellar mass profile.")
            fields["stellar_mass"] = unyt_array(
                integrate_mass(stellar_density, rr), "Msun"
            )

        mdm = fields["total_mass"].copy()
        ddm = fields["total_density"].copy()
        if "density" in fields:
            mdm -= fields["gas_mass"]
            ddm -= fields["density"]
        if "stellar_mass" in fields:
            mdm -= fields["stellar_mass"]
            ddm -= fields["stellar_density"]
        mdm[ddm.v < 0.0] = mdm.max()
        ddm[ddm.v < 0.0] = 0.0

        if ddm.sum() < 0.0 or mdm.sum() < 0.0:
            mylog.warning("The total dark matter mass is either zero or negative!!")
        fields["dark_matter_density"] = ddm
        fields["dark_matter_mass"] = mdm

        if "density" in fields:
            fields["gas_fraction"] = fields["gas_mass"] / fields["total_mass"]
            fields["electron_number_density"] = fields["density"].to(
                "cm**-3", "number_density", mu=mue
            )
            fields["entropy"] = (
                fields["temperature"] * fields["electron_number_density"] ** mtt
            )

        return cls(fields, properties=kwargs)

    def set_rmax(self, r_max):
        """
        Truncate the model at a specified maximal radius.

        Parameters
        ----------
        r_max: float
            The maximal radius at which to truncate.

        Returns
        -------
        ClusterModel
            A new :py:class:`model.ClusterModel` instance with a truncated maximal radius.

        """
        mask = self.fields["radius"].d <= r_max
        fields = {}
        for field in self.fields:
            fields[field] = self.fields[field][mask]
        num_elements = mask.sum()
        return ClusterModel(
            num_elements, fields, dm_virial=self.dm_virial, star_virial=self.star_virial
        )

    def write_model_to_ascii(self, output_filename, in_cgs=False, overwrite=False):
        r"""
        Write the equilibrium model to an ascii text file. Uses
        AstroPy's `QTable` to write the file, so that units are
        included.

        Parameters
        ----------
        output_filename : str
            The file to write the model to.
        in_cgs : bool, optional
            Whether to convert the units to cgs before writing. Default: False.
        overwrite : bool, optional
            Overwrite an existing file with the same name. Default: False.

        """
        from astropy.table import QTable

        fields = {}
        for k, v in self.fields.items():
            if in_cgs:
                if k == "temperature":
                    fd = v.to_equivalent("K", "thermal")
                elif k not in self._keep_units:
                    fd = v.in_cgs()
                else:
                    fd = v
            else:
                fd = v
            fields[k] = fd.to_astropy()
        t = QTable(fields)
        t.meta["comments"] = f"unit_system={'cgs' if in_cgs else 'galactic'}"
        t.write(output_filename, overwrite=overwrite)

    def write_model_to_h5(
        self, output_filename, in_cgs=False, r_min=None, r_max=None, overwrite=False
    ):
        r"""
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : str
            The file to write the model to.
        in_cgs : bool, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : bool, optional
            Overwrite an existing file with the same name. Default False.
        r_min: float, optional
            The minimum radius to write too.
        r_max: float, optional
            The maximum radius to write too.

        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError(
                f"Cannot create {output_filename}. It exists and " f"overwrite=False."
            )
        f = h5py.File(output_filename, "w")
        f.create_dataset("num_elements", data=self.num_elements)
        f.close()

        self.properties["meth"]["in_cgs"] = in_cgs
        self._write_model_attrs(output_filename)

        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(
            self.fields["radius"].d >= r_min, self.fields["radius"].d <= r_max
        )

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
            fd.write_hdf5(output_filename, dataset_name=k, group_name="fields")
        if getattr(self, "_dm_virial", None):
            fd = self.dm_virial.df
            fd.write_hdf5(output_filename, dataset_name="dm_df")
        if getattr(self, "_star_virial", None):
            fd = self.star_virial.df
            fd.write_hdf5(output_filename, dataset_name="star_df")

    def _write_model_attrs(self, output_filename):
        """Write the model attributes to file."""
        import dill as pickle

        with open(f"{output_filename}.properties", "wb") as atf:
            pickle.dump(self.properties, atf)

    def write_model_to_binary(
        self,
        output_filename,
        fields_to_write=None,
        in_cgs=False,
        r_min=None,
        r_max=None,
        overwrite=False,
    ):
        """
        Write the model to unformatted Fortran binary.

        .. attention::

            But why would you want Fortran binary??

        .. warning::

            This procedure will lose all of the metadata from the ``self.gravity`` attribute and the ``self.attrs``
            dictionary. As such, this should only be used in cases where the user can specify those properties manually
            upon reopening the file.

        Parameters
        ----------
        output_filename: str
            The output filename.
        fields_to_write: list
            The list of field names to write.
        in_cgs: bool
            If ``True``, will write in cgs.
        r_min: float
            The minimum radius
        r_max: float
            The maximum radius
        overwrite: bool
            ``True`` will overwrite any existing file with the same path.

        Returns
        -------
        None

        """
        if fields_to_write is None:
            fields_to_write = list(self.fields.keys())
        from scipy.io import FortranFile

        if os.path.exists(output_filename) and not overwrite:
            raise IOError(
                f"Cannot create {output_filename}. It exists and " f"overwrite=False."
            )
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(
            self.fields["radius"].d >= r_min, self.fields["radius"].d <= r_max
        )
        with FortranFile(output_filename, "w") as f:
            f.write_record(self.fields["radius"][mask].size)
            prof_rec = []
            for k in fields_to_write:
                v = self.fields[k]
                if in_cgs:
                    if k == "temperature":
                        fd = v[mask].to_equivalent("K", "thermal")
                    elif k not in self._keep_units:
                        fd = v[mask].in_cgs()
                    else:
                        fd = v[mask]
                else:
                    fd = v[mask]
                prof_rec.append(fd)
            f.write_record(np.array(prof_rec).T)

    def set_field(self, name, value):
        r"""
        Set a field with name `name` to value `value`, which is an `unyt_array`.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, unyt_array):
            raise TypeError("value needs to be an unyt_array")
        if value.size == self.num_elements:
            if name in self.fields:
                mylog.warning("Overwriting field %s." % name)
            self.fields[name] = value
        else:
            raise ValueError(
                f"The length of the array needs to be " f"{self.num_elements} elements!"
            )

    @classmethod
    def from_dens_and_temp(
        cls,
        rmin,
        rmax,
        density,
        temperature,
        stellar_density=None,
        num_points=1000,
        **kwargs,
    ):
        """
        Construct a hydrostatic equilibrium model using gas density
        and temperature profiles.

        Parameters
        ----------
        rmin : float
            Minimum radius of profiles in kpc.
        rmax : float
            Maximum radius of profiles in kpc.
        density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the gas mass density.
        temperature : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the gas temperature.
        stellar_density : :class:`~cluster_generator.radial_profiles.RadialProfile`, optional
            A radial profile describing the stellar mass density, if desired.
        num_points : integer, optional
            The number of points the profiles are evaluated at.

        """
        kwargs = _force_method(kwargs, "from_dens_and_temp")
        kwargs["meth"]["profiles"] = {
            "temperature": temperature,
            "density": density,
            "stellar_density": stellar_density,
        }
        mylog.info("Computing the profiles from density and temperature.")
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["density"] = unyt_array(density(rr), "Msun/kpc**3")
        fields["temperature"] = unyt_array(temperature(rr), "keV")
        fields["pressure"] = fields["density"] * fields["temperature"]
        fields["pressure"] /= mu * mp
        fields["pressure"].convert_to_units("Msun/(Myr**2*kpc)")
        pressure_spline = InterpolatedUnivariateSpline(rr, fields["pressure"].d)
        dPdr = unyt_array(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        fields["gravitational_field"] = dPdr / fields["density"]
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        fields["gas_mass"] = unyt_array(integrate_mass(density, rr), "Msun")
        fields["total_mass"] = (
            -fields["radius"] ** 2 * fields["gravitational_field"] / G
        )
        total_mass_spline = InterpolatedUnivariateSpline(rr, fields["total_mass"].v)
        dMdr = unyt_array(total_mass_spline(rr, nu=1), "Msun/kpc")
        fields["total_density"] = dMdr / (4.0 * np.pi * fields["radius"] ** 2)
        return cls._from_scratch(fields, stellar_density=stellar_density, **kwargs)

    @classmethod
    def from_dens_and_entr(
        cls,
        rmin,
        rmax,
        density,
        entropy,
        stellar_density=None,
        num_points=1000,
        **kwargs,
    ):
        """
        Construct the model from density and entropy.

        Parameters
        ----------
        rmin: float
            The minimum radius at which to compute
        rmax: float
            The maximum radius at which to compute.
        num_points: int
            The number of sample points.
        density: callable
            The gas density profile.
        entropy: callable
            The gas entropy profile.
        stellar_density: callable or unyt_array
            The stellar density profile.

        Returns
        -------
        ClusterModel

        """
        kwargs = _force_method(kwargs, "from_dens_and_entr")
        kwargs["meth"]["profiles"] = {
            "entropy": entropy,
            "density": density,
            "stellar_density": stellar_density,
        }
        n_e = density / (mue * mp * kpc_to_cm**3)
        temperature = entropy * n_e**tt
        return cls.from_dens_and_temp(
            rmin,
            rmax,
            density,
            temperature,
            stellar_density=stellar_density,
            num_points=num_points,
            **kwargs,
        )

    @classmethod
    def from_dens_and_tden(
        cls,
        rmin,
        rmax,
        density,
        total_density,
        stellar_density=None,
        num_points=1000,
        **kwargs,
    ):
        """
        Construct a hydrostatic equilibrium model using gas density
        and total density profiles

        Parameters
        ----------
        rmin : float
            Minimum radius of profiles in kpc.
        rmax : float
            Maximum radius of profiles in kpc.
        density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the gas mass density.
        total_density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the total mass density.
        stellar_density : :class:`~cluster_generator.radial_profiles.RadialProfile`, optional
            A radial profile describing the stellar mass density, if desired.
        num_points : integer, optional
            The number of points the profiles are evaluated at.

        """
        kwargs = _force_method(kwargs, "from_dens_and_tden")
        kwargs["meth"]["profiles"] = {
            "total_density": total_density,
            "density": density,
            "stellar_density": stellar_density,
        }
        mylog.info("Computing the profiles from density and total density.")
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["density"] = unyt_array(density(rr), "Msun/kpc**3")
        fields["total_density"] = unyt_array(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = unyt_array(integrate_mass(total_density, rr), "Msun")
        fields["gas_mass"] = unyt_array(integrate_mass(density, rr), "Msun")
        fields["gravitational_field"] = (
            -G * fields["total_mass"] / (fields["radius"] ** 2)
        )
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        g = fields["gravitational_field"].in_units("kpc/Myr**2").v
        g_r = InterpolatedUnivariateSpline(rr, g)
        dPdr_int = lambda r: density(r) * g_r(r)
        mylog.info("Integrating pressure profile.")
        P = -integrate(dPdr_int, rr)
        dPdr_int2 = lambda r: density(r) * g[-1] * (rr[-1] / r) ** 2
        P -= quad(dPdr_int2, rr[-1], np.inf, limit=100)[0]
        fields["pressure"] = unyt_array(P, "Msun/kpc/Myr**2")
        fields["temperature"] = fields["pressure"] * mu * mp / fields["density"]
        fields["temperature"].convert_to_units("keV")

        return cls._from_scratch(fields, stellar_density=stellar_density, **kwargs)

    @classmethod
    def no_gas(
        cls, rmin, rmax, total_density, stellar_density=None, num_points=1000, **kwargs
    ):
        """
        Initialize a :py:class:`model.ClusterModel` which is composed only of collisionless species.

        Parameters
        ----------
        rmin : float
            Minimum radius of profiles in kpc.
        rmax : float
            Maximum radius of profiles in kpc.
        total_density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the total mass density.
        stellar_density : :class:`~cluster_generator.radial_profiles.RadialProfile`, optional
            A radial profile describing the stellar mass density, if desired.
        num_points : integer, optional
            The number of points the profiles are evaluated at.
        kwargs:
            Additional keywords to pass to the properties of the resulting instance.

        Returns
        -------
        :py:class:`model.ClusterModel`

        """
        kwargs = _force_method(kwargs, "no_gas")
        kwargs["meth"]["profiles"] = {
            "total_density": total_density,
            "stellar_density": stellar_density,
        }
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["total_density"] = unyt_array(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = unyt_array(integrate_mass(total_density, rr), "Msun")
        fields["gravitational_field"] = (
            -G * fields["total_mass"] / (fields["radius"] ** 2)
        )
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        return cls._from_scratch(fields, stellar_density=stellar_density, **kwargs)

    def find_field_at_radius(self, field, r):
        """
        Find the value of a *field* in the profiles
        at radius *r*.
        """
        return unyt_array(np.interp(r, self["radius"], self[field]), self[field].units)

    def check_hse(self):
        r"""
        Determine the deviation of the model from hydrostatic equilibrium.

        Returns
        -------
        chk : NumPy array
            An array containing the relative deviation from hydrostatic
            equilibrium as a function of radius.

        """
        if "pressure" not in self.fields:
            raise RuntimeError("This ClusterModel contains no gas!")
        rr = self.fields["radius"].v
        pressure_spline = InterpolatedUnivariateSpline(rr, self.fields["pressure"].v)
        dPdx = unyt_array(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        rhog = self.fields["density"] * self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info(
            "The maximum relative deviation of this profile from "
            "hydrostatic equilibrium is %g",
            np.abs(chk).max(),
        )
        return chk

    def check_dm_virial(self):
        """Equivalent to ``self.dm_virial.check_virial``"""
        return self.dm_virial.check_virial()

    def check_star_virial(self):
        """Equivalent to ``self.star_virial.check_virial``"""
        if self._star_virial is None:
            raise RuntimeError(
                "Cannot check the virial equilibrium of "
                "the stars because there are no stars in "
                "this model!"
            )
        return self.star_virial.check_virial()

    def set_magnetic_field_from_beta(self, beta, gaussian=True):
        """
        Set a magnetic field radial profile from
        a plasma beta parameter, assuming beta = p_th/p_B.
        The field can be set in Gaussian or Lorentz-Heaviside
        (dimensionless) units.

        Parameters
        ----------
        beta : float
            The ratio of the thermal pressure to the
            magnetic pressure.
        gaussian : boolean, optional
            Set the field in Gaussian units such that
            p_B = B^2/(8*pi), otherwise p_B = B^2/2.
            Default: True

        """
        B = np.sqrt(2.0 * self["pressure"] / beta)
        if gaussian:
            B *= np.sqrt(4.0 * np.pi)
        B.convert_to_units("gauss")
        self.set_field("magnetic_field_strength", B)

    def set_magnetic_field_from_density(self, B0, eta=2.0 / 3.0, gaussian=True):
        """
        Set a magnetic field radial profile assuming it is proportional
        to some power of the density, usually 2/3. The field can be set
        in Gaussian or Lorentz-Heaviside (dimensionless) units.

        Parameters
        ----------
        B0 : float
            The central magnetic field strength in units of
            gauss.
        eta : float, optional
            The power of the density which the field is
            proportional to. Default: 2/3.
        gaussian : boolean, optional
            Set the field in Gaussian units such that
            p_B = B^2/(8*pi), otherwise p_B = B^2/2.
            Default: True

        """
        B0 = ensure_ytquantity(B0, "gauss")
        B = B0 * (self["density"] / self["density"][0]) ** eta
        if not gaussian:
            B /= np.sqrt(4.0 * np.pi)
        self.set_field("magnetic_field_strength", B)

    def generate_tracer_particles(
        self, num_particles, r_max=None, sub_sample=1, prng=None
    ):
        """
        Generate a set of tracer particles based on the gas distribution.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii, densities, and energies which will then be
            repeated to fill the required number of particles. Default: 1,
            which means no sub-sampling.
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.

        """
        from cluster_generator.utils import parse_prng

        prng = parse_prng(prng)
        mylog.info("We will be assigning %d tracer particles.", num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(
            self["radius"].d,
            self["gas_mass"].d,
            num_particles_sub,
            r_max=r_max,
            prng=prng,
        )

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_particles))
        phi = 2.0 * np.pi * prng.uniform(size=num_particles)

        fields = OrderedDict()

        fields["tracer", "particle_position"] = unyt_array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ],
            "kpc",
        ).T

        fields["tracer", "particle_velocity"] = unyt_array(
            np.zeros(fields["tracer", "particle_position"].shape), "kpc/Myr"
        )

        fields["tracer", "particle_mass"] = unyt_array(np.zeros(num_particles), "Msun")

        return ClusterParticles("tracer", fields)

    def generate_gas_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of gas particles in hydrostatic equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii, densities, and energies which will then be
            repeated to fill the required number of particles. Default: 1,
            which means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.

        """
        from cluster_generator.utils import parse_prng

        prng = parse_prng(prng)
        mylog.info("We will be assigning %d gas particles.", num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(
            self["radius"].d,
            self["gas_mass"].d,
            num_particles_sub,
            r_max=r_max,
            prng=prng,
        )

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_particles))
        phi = 2.0 * np.pi * prng.uniform(size=num_particles)

        fields = OrderedDict()

        fields["gas", "particle_position"] = unyt_array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ],
            "kpc",
        ).T

        mylog.info("Compute particle thermal energies, densities, and masses.")

        e_arr = 1.5 * self.fields["pressure"] / self.fields["density"]
        get_energy = InterpolatedUnivariateSpline(self.fields["radius"], e_arr)

        if sub_sample > 1:
            energy = np.tile(get_energy(radius_sub), sub_sample)[:num_particles]
        else:
            energy = get_energy(radius)

        fields["gas", "thermal_energy"] = unyt_array(energy, "kpc**2/Myr**2")
        fields["gas", "particle_mass"] = unyt_array(
            [mtot / num_particles] * num_particles, "Msun"
        )

        get_density = InterpolatedUnivariateSpline(
            self.fields["radius"], self.fields["density"]
        )

        if sub_sample > 1:
            density = np.tile(get_density(radius_sub), sub_sample)[:num_particles]
        else:
            density = get_density(radius)

        fields["gas", "density"] = unyt_array(density, "Msun/kpc**3")

        mylog.info("Set particle velocities to zero.")

        fields["gas", "particle_velocity"] = unyt_array(
            np.zeros((num_particles, 3)), "kpc/Myr"
        )

        if compute_potential:
            energy_spline = InterpolatedUnivariateSpline(
                self["radius"].d, -self["gravitational_potential"]
            )
            phi = -energy_spline(radius_sub)
            if sub_sample > 1:
                phi = np.tile(phi, sub_sample)
            fields["gas", "particle_potential"] = unyt_array(phi, "kpc**2/Myr**2")

        return ClusterParticles("gas", fields)

    def generate_dm_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of dark matter particles in virial equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii and velocities which will then be repeated
            to fill the required number of particles. Default: 1, which
            means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.

        Returns
        -------
        particles : :class:`~cluster_generator.particles.ClusterParticles`
            A set of dark matter particles.

        """
        return self.dm_virial.generate_particles(
            num_particles,
            r_max=r_max,
            sub_sample=sub_sample,
            compute_potential=compute_potential,
            prng=prng,
        )

    def generate_star_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of star particles in virial equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii and velocities which will then be repeated
            to fill the required number of particles. Default: 1, which
            means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.

        Returns
        -------
        particles : :class:`~cluster_generator.particles.ClusterParticles`
            A set of star particles.

        """
        return self.star_virial.generate_particles(
            num_particles,
            r_max=r_max,
            sub_sample=sub_sample,
            compute_potential=compute_potential,
            prng=prng,
        )

    def plot(self, field, r_min=None, r_max=None, fig=None, ax=None, **kwargs):
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
        ax.loglog(self["radius"], self[field], **kwargs)
        ax.set_xlim(r_min, r_max)
        ax.set_xlabel("Radius (kpc)")
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        return fig, ax

    @_enforce_style
    def panel_plot(
        self,
        fields="all",
        r_min=None,
        r_max=None,
        fig=None,
        axes=None,
        aspect_ratio=1,
        base_length=3,
        **kwargs,
    ):
        """
        Plot all of the selected fields in a grid of axes. Useful for checking for non-physical issues and getting
        a general impression of the generated model.

        Parameters
        ----------
        fields: list or str
            The fields to include in the plot. If ``fields="all"``, then all of the fields available will be plotted. Otherwise, ``fields`` should
            be a list of field names.
        r_min: float or int, optional
            The minimum radius (kpc) from which to plot. Determined if not specified.
        r_max: float or int, optional
            The maximum radius (kpc) from which to plot. Determined if not specified.
        fig: :py:class:`~matplotlib.figure.Figure`, optional
            A :py:class:`~matplotlib.figure.Figure` instance onto which to place the plots. If not specified, a fresh one will be generated.

            .. warning::

                If a figure is specified but the axes are not specified, then this will overwrite / cover up any pre-existing
                plots on those axes.
        axes: dict of str: :py:class:`~matplotlib.axes.Axes`, optional
            The axes onto which the plots should be drawn. These should be organized as a dictionary with keys corresponding to the
            field name belonging to the plot and the corresponding axes on which that plot is drawn. If not specified, a grid is
            generated on the figure and subdivided to facilitate the drawing.

        aspect_ratio: float, optional
            The aspect ratio of the individual subplots.

        base_length: float, optional
            The base length of each plot. Used to determine rest of the geometry.
        **kwargs
            Additional keyword arguments which are to be passed directly to :py:func:`matplotlib.pyplot.plot`. There are several formatting
            options for these keyword arguments and the behavior depends on their format.

            - If the kwarg values are :py:class:`dict`, then they should have keys corresponding to the different fields and
              those kwargs will be passed to the field plot. If a key is missing, then no kwargs are passed for that
              setting.
            - If kwarg values are not dictionaries, they will be applied uniformly across all of the subplots.

        Returns
        -------
        :py:class:`~matplotlib.figure.Figure`
            The figure corresponding to the plot.
        :py:class:`~matplotlib.axes.Axes`
            The axes dictionary corresponding to the plot

        """
        import pathlib as pt
        from itertools import product

        import matplotlib.pyplot as plt

        from cluster_generator.numalgs import _closest_factors

        # Loading the plot defaults
        _config_directory = os.path.join(
            pt.Path(__file__).parents[0], "bin", "resources", "plot_defaults.yaml"
        )

        with open(_config_directory, "r") as f:
            plt_defaults = yaml.load(f, yaml.FullLoader)

        if fields == "all":
            fields = list(self.fields.keys())

        if r_min is None:
            r_min = np.amin(self["radius"].d)
        if r_max is None:
            r_max = np.amax(self["radius"].d)
        if fig is None:
            fig = plt.figure()

        if axes is None:
            axes = {}
            r, c = _closest_factors(len(fields))

            # computing the figure behavior
            h = (1 / (1 - 0.16)) * r * aspect_ratio * base_length
            w = ((1 + 0.2) / (1 - 0.16)) * c * base_length
            fig.set_figwidth(w)
            fig.set_figheight(h)
            fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.05)

            gridspec = fig.add_gridspec(r, c)
            gspec_it = product(range(0, r), range(0, c))

            for gi, fi in zip(gspec_it, fields):
                ri, ci = gi
                axes[fi] = fig.add_subplot(gridspec[ri, ci])
        else:
            assert all(i in axes for i in fields), "Some fields do not have axes."

        for field, ax in axes.items():
            tmp_kwargs = {
                k: (v[field] if isinstance(v, dict) else v) for k, v in kwargs.items()
            }

            if field in plt_defaults:
                for k, v in plt_defaults[field]["kwargs"]:
                    if k not in tmp_kwargs:
                        tmp_kwargs[k] = v

                revision = (
                    plt_defaults[field]["revision"]
                    if plt_defaults[field]["revision"] is not None
                    else 1
                )
            else:
                revision = 1

            ax.loglog(
                self["radius"][
                    np.where((r_min < self["radius"].d) & (r_max > self["radius"].d))
                ],
                revision
                * self[field][
                    np.where((r_min < self["radius"].d) & (r_max > self["radius"].d))
                ],
                **tmp_kwargs,
            )
            if field in plt_defaults:
                if "label" in plt_defaults[field]:
                    ax.set_ylabel(
                        plt_defaults[field]["label"]
                        % {"unit": self[field].units.latex_representation()}
                    )

                if "scale" in plt_defaults[field]:
                    ax.set_yscale(plt_defaults[field]["scale"])
            else:
                ax.set_ylabel(
                    r"%(field)s / $\left[%(unit)s\right]$"
                    % {"field": field, "unit": self[field].units.latex_representation()}
                )

            ax.set_xlabel(
                r"Radius / $\left[%(unit)s\right]$"
                % {"unit": self["radius"].units.latex_representation()}
            )
        return fig, axes

    def mass_in_radius(self, radius):
        """Determine the mass within a given radius."""
        masses = {}
        r = self.fields["radius"].to_value("kpc")
        for mtype in ["total", "gas", "dark_matter", "stellar"]:
            if f"{mtype}_mass" in self.fields:
                masses[mtype] = self.fields[f"{mtype}_mass"][r < radius][-1]
        return masses

    def find_radius_for_density(self, density):
        """Determine the radius at which the model reaches the specified density"""
        density = ensure_ytquantity(density, "Msun/kpc**3").value
        r = self.fields["radius"].to_value("kpc")[::-1]
        d = self.fields["density"].to_value("Msun/kpc**3")[::-1]
        return unyt_quantity(np.interp(density, d, r), "kpc")

    def create_dataset(
        self, domain_dimensions, box_size=None, left_edge=None, velocity=None, **kwargs
    ):
        """
        .. warning::

            This method can be memory intensive. We suggest being conservative in your choice of domain size to
            begin in order to avoid OOM issues. For reference, a domain dimension of ``[500,500,500]`` will take appox.
            3Gb / field.

        Create an in-memory, uniformly gridded dataset in 3D using yt by
        placing the cluster into a box.

        Parameters
        ----------
        domain_dimensions : 3-tuple of ints
            The number of cells on a side for the domain.
        box_size : float, optional
            The size of the box in kpc. If not specified, will default to the size of the cluster.
        left_edge : array_like, optional
            The minimum coordinate of the box in all three dimensions,
            in kpc. Default: None, which means the left edge will
            be [-r,-r,-r].
        velocity: unyt_array, optional
            The velocity of the cluster relative to the frame of the box. Default is 0. If specified, must be a 1X3 array.

        """
        import gc

        from yt.loaders import load_uniform_grid

        from cluster_generator.utils import build_yt_dataset_fields, mylog

        mylog.info(f"Loading yt dataset of {self}...")

        if box_size is None:
            box_size = 2 * np.amax(self["radius"].d)

        # Dealing with the left edge of the domain.
        if left_edge is None:
            left_edge = -np.amax(box_size / 2) * np.ones(3)
        left_edge = np.array(left_edge)

        # Building the boundary box
        bbox = [
            [left_edge[0], left_edge[0] + box_size],
            [left_edge[1], left_edge[1] + box_size],
            [left_edge[2], left_edge[2] + box_size],
        ]

        # Creating the underlying meshgrid
        try:
            x, y, z = np.mgrid[
                bbox[0][0] : bbox[0][1] : domain_dimensions[0] * 1j,
                bbox[1][0] : bbox[1][1] : domain_dimensions[1] * 1j,
                bbox[2][0] : bbox[2][1] : domain_dimensions[2] * 1j,
            ]
        except MemoryError as err:
            raise MemoryError(
                f"Failed to allocate memory for the grid. Error msg = {err.__str__()}."
            )

        gc.collect()

        data = build_yt_dataset_fields(
            [x, y, z],
            [self],
            domain_dimensions,
            unyt_array([[0, 0, 0]], "kpc"),
            velocity if velocity is not None else unyt_array([[0, 0, 0]], "kpc/Myr"),
        )

        gc.collect()
        return load_uniform_grid(
            data,
            domain_dimensions,
            length_unit="kpc",
            bbox=np.array(bbox),
            mass_unit="Msun",
            time_unit="Myr",
            **kwargs,
        )


# This is only for backwards-compatibility
class HydrostaticEquilibrium(ClusterModel):
    """Equivalent to :py:class:`ClusterModel`"""

    pass


def _force_method(dictionary, meth):
    if "meth" in dictionary:
        if "method" in dictionary["meth"]:
            pass
        else:
            dictionary["meth"]["method"] = meth
    else:
        dictionary["meth"] = {"method": meth}

    return dictionary
