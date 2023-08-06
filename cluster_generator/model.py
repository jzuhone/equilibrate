"""
Complete models of galaxy clusters including hydrostatic equilibrium.
"""
import os
from collections import OrderedDict

import astropy.units
import h5py
import numpy as np
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array

from cluster_generator.particles import \
    ClusterParticles
from cluster_generator.utils import \
    integrate, mylog, integrate_mass, \
    mp, G, generate_particle_radii, mu, mue, \
    ensure_ytquantity, kpc_to_cm
from cluster_generator.virial import \
    VirialEquilibrium
from cluster_generator.gravity import Potential,_default_a_0,_default_interpolation_function
from scipy.optimize import fsolve
import astropy

tt = 2.0 / 3.0
mtt = -tt
ft = 5.0 / 3.0
tf = 3.0 / 5.0
mtf = -tf
gamma = ft
et = 8.0 / 3.0
te = 3.0 / 8.0


class ClusterModel(Potential):
    r"""
    The ``ClusterModel`` class is a comprehensive representation of the cluster being modeled and can be used to generate
    accurate initial conditions. The class is predicated on a fixed number of sample radii in the cluster.

    Parameters
    ----------
    num_elements: int
        The number of elements included. This is equivalent to the number of radii at which the model is sampled.
    fields: dict[str,unyt_array]
        The fields to attribute to the ``ClusterModel``.
    attrs: dict, default = None
        Additional attributes to pass to the cluster model. The available attributes are

        +---------------------+------------------------------------------+----------------+---------------------------------------------------+
        | Attribute Name      | Description                              | Types          | Default                                           |
        +=====================+==========================================+================+===================================================+
        | ``interp_function`` | The MOND specific interpolation function | ``callable``   | :math:`\frac{x}{x+1}`                             |
        +---------------------+------------------------------------------+----------------+---------------------------------------------------+
        | ``a_0``             | The MOND acceleration constant.          | ``unyt float`` | :math:`1.2\times 10^{-10} \mathrm{\frac{m}{s^2}}` |
        +---------------------+------------------------------------------+----------------+---------------------------------------------------+
        | ``change_T``        | See Cluster Models documentation         | ``bool``       | ``False``                                         |
        +---------------------+------------------------------------------+----------------+---------------------------------------------------+

    gravity: str
        The gravity theory to apply. Options are ``["Newtonian,AQUAL,QUMOND]``.

    Notes
    -----

    - ``__getitem__`` and ``__contains__`` are aliased down to ``self.fields``. There is no ``__setitem__``, so index
      / key assignment cannot be done. use ``ClusterModel.set_field()`` instead.
    """
    #  Class Variables
    # ----------------------------------------------------------------------------------------------------------------- #
    #: The default included fields that can be accessed.
    default_fields = ["density", "temperature", "pressure", "total_density",
                      "gravitational_potential", "gravitational_field",
                      "total_mass", "gas_mass", "dark_matter_mass",
                      "dark_matter_density", "stellar_density", "stellar_mass"]

    _keep_units = ["entropy", "electron_number_density",
                   "magnetic_field_strength"]

    _dm_virial = None
    _star_virial = None

    #  Dunder Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    def __init__(self, fields,attrs=None,gravity="Newtonian"):
        #  Setting basic attributes
        # ------------------------------------------------------------------------------------------------------------ #

        # - Super Initialization - #
        super().__init__(fields,gravity,attrs)

    def __repr__(self):
        return f"ClusterModel object; gravity={self.gravity}"

    def __str__(self):
        return f"ClusterModel object; gravity={self.gravity}, fields={list(self.fields.keys())}"

    #  Properties
    # ----------------------------------------------------------------------------------------------------------------- #

    @property
    def dm_virial(self):
        #TODO: THIS GETS CHANGED
        if self._dm_virial is None:
            self._dm_virial = VirialEquilibrium(self, "dark_matter",gravity=self.gravity) #TODO: this needs local maxwellian check
        return self._dm_virial

    @property
    def star_virial(self):
        #TODO: THIS GETS CHANGED
        if self._star_virial is None and "stellar_density" in self:
            self._star_virial = VirialEquilibrium(self, "stellar",gravity=self.gravity) #TODO: this check needs local maxwellian check.
        return self._star_virial

    #  Class Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    @classmethod
    def from_arrays(cls, fields,**kwargs):
        """
        Initialize the ``ClusterModel`` from ``fields`` alone.

        Parameters
        ----------
        fields: dict[str:unyt_array]
            The fields from which to generate the model.

            .. admonition:: development note

                The ``fields`` parameter must be self consistent in its definition, and must contain a ``radius`` key, from
                which the class assesses the number of elements.
        kwargs:
            Additional values to pass to the initialization method. This should include the gravity type and any attributes.

        Returns
        -------
        ClusterModel

        Notes
        -----
        - Equivalent to ``ClusterModel(fields["radius"].size,fields)``.

        Examples
        --------
        >>> from cluster_generator.tests.utils import generate_mdr_potential
        >>> import matplotlib.pyplot as plt
        >>> m,d,r = generate_mdr_potential()
        >>> fields = {"total_mass":m,"total_density":d,"radius":r}
        >>> model = ClusterModel.from_arrays(fields)
        >>> print(model.fields.keys())
        dict_keys(['total_mass', 'total_density', 'radius', 'gravitational_potential'])
        >>> _ = model.pot # Generate the potential
        >>> model.plot("gravitational_potential")

        .. image:: ../_images/model/from_arrays_plot.png
        """
        if "stellar_density" in kwargs:
            fields["stellar_density"] = kwargs["stellar_density"]
            del kwargs["stellar_density"]

        return cls(fields,**kwargs)

    @classmethod
    def from_h5_file(cls, filename, r_min=None, r_max=None):
        r"""
        Generate an equilibrium model from an HDF5 file.

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.
        r_min: float, optional
            The minimum radius
        r_max: float, optional
            The maximal radius
        """
        from cluster_generator.virial import VirialEquilibrium

        # - Grabbing base data -#
        with h5py.File(filename, "r") as f:
            fnames = list(f['fields'].keys())
            get_dm_virial = 'dm_df' in f #TODO: CHECK THIS
            get_star_virial = 'star_df' in f #TODO:CHECK THIS

            # Grabbing additional attributes #
            _attrs = dict(f.attrs)

        fields = OrderedDict()
        for field in fnames:  # -> converting fields to unyt_arrays.
            a = unyt_array.from_hdf5(filename, dataset_name=field,
                                     group_name="fields")
            fields[field] = unyt_array(a.d, str(a.units))
            if field not in cls._keep_units:  # --> using data conversion.
                fields[field].convert_to_base("galactic")

        # - Determining rmin / rmax and masking. -#
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = fields["radius"][-1].d * 2
        mask = np.logical_and(fields["radius"].d >= r_min,
                              fields["radius"].d <= r_max)
        for field in fnames:
            fields[field] = fields[field][mask]
        num_elements = mask.sum()

        # - Managing gravity - #
        if "gravity" in _attrs:
            _grav = _attrs["gravity"]
            del _attrs["gravity"]
        else:
            _grav = "Newtonian"

        # - Creating the model - #
        model = cls(fields, attrs=_attrs,gravity=_grav)

        # - Virializing -#
        if get_dm_virial: #TODO:CHECK
            mask = np.logical_and(fields["radius"].d >= r_min,
                                  fields["radius"].d <= r_max)
            df = unyt_array.from_hdf5(
                filename, dataset_name="dm_df")[mask]
            model._dm_virial = VirialEquilibrium(
                model, ptype="dark_matter", df=df)

        if get_star_virial: #TODO:CHECK
            mask = np.logical_and(fields["radius"].d >= r_min,
                                  fields["radius"].d <= r_max)
            df = unyt_array.from_hdf5(
                filename, dataset_name="star_df")[mask]
            model._star_virial = VirialEquilibrium(
                model, ptype="stellar", df=df)

        return model

    @classmethod
    def _from_scratch(cls, fields, stellar_density=None,gravity="Newtonian",attrs=None):
        #  Sanity check
        # ------------------------------------------------------------------------------------------------------------ #
        mylog.debug("Initializing ClusterModel using ._from_scratch().")

        _required_fields = ["radius","total_density","total_mass"]

        for field in _required_fields:
            if field not in fields:
                ValueError(f"Failed to find required field {field} for generation using _from_scratch.")

        #  Pulling data
        # ------------------------------------------------------------------------------------------------------------ #
        rr = fields["radius"].d

        # Standardizing Construction
        # ----------------------------------------------------------------------------------------------------------------- #
        #- Gas mass integration -#
        if "density" in fields and "gas_mass" not in fields:
            mylog.info("\tIntegrating gas mass profile.")
            m0 = fields["density"].d[0] * rr[0] ** 3
            fields["gas_mass"] = unyt_array(
                (4.0/3.0) * np.pi * cumtrapz(fields["density"] * rr * rr,
                                       x=rr, initial=0.0) + m0, "Msun")

        #- Managing the stellar component
        if stellar_density is not None:
            mylog.info("\tIntegrating stellar mass profile.")
            fields["stellar_density"] = unyt_array(stellar_density(rr),
                                                   "Msun/kpc**3")
            fields["stellar_mass"] = unyt_array(
                integrate_mass(stellar_density, rr), "Msun")

        mylog.info("\tDetermining the halo component.")
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
            fields["electron_number_density"] = \
                fields["density"].to("cm**-3", "number_density", mu=mue)
            fields["entropy"] = \
                fields["temperature"] * fields["electron_number_density"] ** mtt

        obj = cls(fields,attrs=attrs,gravity=gravity)
        _ = obj.pot
        return obj

    @classmethod
    def from_dens_and_temp(cls, density, temperature,rmin,rmax,num_points,
                           stellar_density=None, gravity="Newtonian",attrs=None):
        """
        Computes the ``ClusterModel`` from gas density and temperature.

        Parameters
        ----------
        rmin: float
            The minimum radius at which to compute
        rmax: float
            The maximum radius at which to compute.
        num_points: int
            The number of points to sample form.
        density: callable
            The gas density profile.
        temperature: callable
            The gas temperature profile.

            .. warning::

                The temperature profile must be in units of ``keV``.

        stellar_density: callable or unyt_array
            The stellar density profile.
        gravity: str
            The gravity type to use.
        attrs: dict
            Additional attributes to pass to ``ClusterModel``. See documentation for more details.
        Returns
        -------
        ClusterModel

        """
        mylog.info(f"Computing the profiles from density and temperature. Gravity={gravity}")

        #  Building the radius array
        # ------------------------------------------------------------------------------------------------------------ #
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)

        #  Building arrays
        # ----------------------------------------------------------------------------------------------------------------- #
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["density"] = unyt_array(density(rr), "Msun/kpc**3")
        fields["temperature"] = unyt_array(temperature(rr), "keV")

        # - Deriving the pressure field - #
        fields["pressure"] = (fields["density"] * fields["temperature"]) / (mu * mp)
        fields["pressure"].convert_to_units("Msun/(Myr**2*kpc)")
        pressure_spline = InterpolatedUnivariateSpline(rr, fields["pressure"].d)

        # - Deriving acceleration - #
        dPdr = unyt_array(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        fields["gravitational_field"] = dPdr / fields["density"]
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        # - Integrating to get gas mass - #
        fields["gas_mass"] = unyt_array(integrate_mass(density, rr), "Msun")

        fields["total_mass"] = _compute_total_mass(fields,gravity=gravity)

        total_mass_spline = InterpolatedUnivariateSpline(rr,
                                                         fields["total_mass"].v)
        dMdr = unyt_array(total_mass_spline(rr, nu=1), "Msun/kpc")
        fields["total_density"] = dMdr / (4. * np.pi * fields["radius"] ** 2)
        return cls._from_scratch(fields, stellar_density=stellar_density,attrs=attrs)

    @classmethod
    def from_dens_and_entr(cls, density, entropy,rmin,rmax,num_points,
                           stellar_density=None,attrs=None):
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
        gravity: str
            The gravity type to use.
        attrs: dict
            Additional attributes to pass to ``ClusterModel``. See documentation for more details.

        Returns
        -------
        ClusterModel

        """
        n_e = density / (mue * mp * kpc_to_cm ** 3)
        temperature = entropy * n_e ** tt
        return cls.from_dens_and_temp(density, temperature,rmin,rmax,num_points,
                                      stellar_density=stellar_density,gravity="Newtonian",attrs=attrs)

    @classmethod
    def from_dens_and_tden(cls, rmin, rmax, density, total_density,
                           stellar_density=None, num_points=1000,gravity="Newtonian",attrs=None):
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
        attrs: dict
            Additional attributes to pass to ``ClusterModel``. See documentation for more details.
        gravity: str
            The gravity theory to use
        """
        mylog.info(f"Computing the profiles from density and total density. Gravity={gravity}")

        #  Pulling parameters
        # ------------------------------------------------------------------------------------------------------------ #
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)

        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["density"] = unyt_array(density(rr), "Msun/kpc**3")
        fields["total_density"] = unyt_array(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = unyt_array(integrate_mass(total_density, rr),
                                          "Msun")
        fields["gas_mass"] = unyt_array(integrate_mass(density, rr), "Msun")

        # - Generating the output object - #
        obj = cls.from_arrays(fields,stellar_density=stellar_density,gravity=gravity,attrs=attrs)

        # - Getting the potential - #
        _ = obj.pot
        _potential_spine = InterpolatedUnivariateSpline(rr,obj.pot.to("kpc**2/Myr**2").d)
        obj["gravitational_field"] = unyt_array(-_potential_spine(rr,1),units="kpc/Myr**2")


        # - Computing temperature - #
        g = obj["gravitational_field"].in_units("kpc/Myr**2").v
        g_r = InterpolatedUnivariateSpline(rr, g)
        dPdr_int = lambda r: density(r) * g_r(r)
        mylog.info("Integrating pressure profile.")
        P = -integrate(dPdr_int, rr)
        dPdr_int2 = lambda r: density(r) * g[-1] * (rr[-1] / r) ** 2
        P -= quad(dPdr_int2, rr[-1], np.inf, limit=100)[0]
        fields["pressure"] = unyt_array(P, "Msun/kpc/Myr**2")
        fields["temperature"] = fields["pressure"] * mu * mp / fields["density"]
        fields["temperature"].convert_to_units("keV")

        return cls._from_scratch(fields, stellar_density=stellar_density,attrs=attrs,gravity=gravity)

    @classmethod
    def no_gas(cls, rmin, rmax, total_density, stellar_density=None,
               num_points=1000):
        """
        Generates the cluster without gas.

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
        gravity: str
            The gravity theory to use
        attrs: dict
            Additional attributes to pass to ``ClusterModel``. See documentation for more details.

        Returns
        -------
        ClusterModel
        """

        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["total_density"] = unyt_array(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = unyt_array(integrate_mass(total_density, rr),
                                          "Msun")

        return cls._from_scratch(fields, stellar_density=stellar_density,gravity="Newtonian")

    #  Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    def keys(self):
        return self.fields.keys()
    def set_rmax(self, r_max):
        mask = self.fields["radius"].d <= r_max
        fields = {}
        for field in self.fields:
            fields[field] = self.fields[field][mask]
        num_elements = mask.sum()
        return ClusterModel(num_elements, fields, dm_virial=self.dm_virial,
                            star_virial=self.star_virial)


    def write_model_to_ascii(self, output_filename, in_cgs=False,
                             overwrite=False):
        r"""
        Write the equilibrium model to an ascii text file. Uses
        AstroPy's `QTable` to write the file, so that units are
        included.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default: False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default: False.

        Examples
        --------
        >>> # - Imports - #
        >>> from cluster_generator.tests.utils import generate_model
        >>> import tempfile
        >>> # Generating the model #
        >>> mdl = generate_model(gravity="Newtonian")
        >>> with tempfile.TemporaryDirectory() as temp_dir:
        ...     mdl.write_model_to_ascii(os.path.join(temp_dir,"model.h5"),overwrite=True)
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

            # - Checking for dimension issues - #
            if str(fd.units) == "dimensionless":
                fields[k] = fd.d * astropy.units.dimensionless_unscaled
            else:
                fields[k] = fd.to_astropy()
        t = QTable(fields)
        t.meta['comments'] = f"unit_system={'cgs' if in_cgs else 'galactic'}"

        # Adding all of the additional info #
        t.meta["gravity"] = self.gravity
        t.meta["attrs"] = self.attrs

        t.write(output_filename, overwrite=overwrite,serialize_meta=True)

    def write_model_to_h5(self, output_filename, in_cgs=False, r_min=None,
                          r_max=None, overwrite=False):
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

        Examples
        --------
        >>> # - Imports - #
        >>> from cluster_generator.tests.utils import generate_model
        >>> import tempfile
        >>> # Generating the model #
        >>> mdl = generate_model(gravity="Newtonian")
        >>> with tempfile.TemporaryDirectory() as temp_dir:
        ...     mdl.write_model_to_h5(os.path.join(temp_dir,"model.h5"),overwrite=True)
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError(f"Cannot create {output_filename}. It exists and "
                          f"overwrite=False.")
        f = h5py.File(output_filename, "w")
        f.create_dataset("num_elements", data=self.num_elements)
        f.attrs["unit_system"] = "cgs" if in_cgs else "galactic"
        f.close()
        if r_min is None:
            r_min = 0.0
        if r_max is None:
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
        if getattr(self, "_dm_virial", None):
            fd = self.dm_virial.df
            fd.write_hdf5(output_filename, dataset_name="dm_df")
        if getattr(self, "_star_virial", None):
            fd = self.star_virial.df
            fd.write_hdf5(output_filename, dataset_name="star_df")

    def write_model_to_binary(self, output_filename, fields_to_write=None,
                              in_cgs=False, r_min=None, r_max=None,
                              overwrite=False):
        """
        writes the model to unformatted Fortran binary.

        .. attention::

            But why would you want Fortan binary??

        .. warning::

            This proceedure will lose all of the metadata from the ``self.gravity`` attribute and the ``self.attrs``
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

        Examples
        --------
        >>> # - Imports - #
        >>> from cluster_generator.tests.utils import generate_model
        >>> import tempfile
        >>> # Generating the model #
        >>> mdl = generate_model(gravity="Newtonian")
        >>> with tempfile.TemporaryDirectory() as temp_dir:
        ...     mdl.write_model_to_binary(os.path.join(temp_dir,"model.h5"),overwrite=True)
        """
        if fields_to_write is None:
            fields_to_write = list(self.fields.keys())
        from scipy.io import FortranFile
        if os.path.exists(output_filename) and not overwrite:
            raise IOError(f"Cannot create {output_filename}. It exists and "
                          f"overwrite=False.")
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(self.fields["radius"].d >= r_min,
                              self.fields["radius"].d <= r_max)
        with FortranFile(output_filename, 'w') as f:
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
            raise ValueError(f"The length of the array needs to be "
                             f"{self.num_elements} elements!")

    def find_field_at_radius(self, field, r):
        """
        Find the value of a *field* in the profiles
        at radius *r*.
        """
        return unyt_array(np.interp(r, self["radius"], self[field]),
                          self[field].units)

    def check_hse(self):
        r"""
        Determine the deviation of the model from hydrostatic equilibrium.

        Returns
        -------
        chk : NumPy array
            An array containing the relative deviation from hydrostatic
            equilibrium as a function of radius.

        Notes
        -----
        Recall that the hydrostatic vector :math:`\Gamma` is defined such that

        .. math::

            \Gamma = -\frac{\nabla P}{\rho_g} = \nabla \Phi,

        Thus, in hydrostatic equilibrium, :math:`\nabla P + \nabla \Phi \rho_g = 0`. To produce a scale invariant quantity, we instead
        report

        .. math::

            \alpha = \frac{\nabla P + \nabla \Phi \rho_g}{\nabla \Phi \rho_g}.

        Examples
        --------
        >>> # - Imports - #
        >>> from cluster_generator.tests.utils import generate_model
        >>> import numpy as np
        >>> from numpy.testing import assert_almost_equal
        >>> # Generating the model #
        >>> mdl = generate_model(gravity="Newtonian")
        >>> # Checking HSE
        >>> assert_almost_equal(np.amax(mdl.check_hse().d),0,decimal=3)
        """
        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if "pressure" not in self.fields:
            raise RuntimeError("This ClusterModel contains no gas!")
        #  Pulling necessary fields
        # ----------------------------------------------------------------------------------------------------------------- #
        rr = self.fields["radius"].v
        pressure_spline = InterpolatedUnivariateSpline(
            rr, self.fields["pressure"].v)
        dPdx = unyt_array(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        rhog = self.fields["density"] * self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info("The maximum relative deviation of this profile from "
                   "hydrostatic equilibrium is %g", np.abs(chk).max())
        return chk

    def check_dm_virial(self):
        return self.dm_virial.check_virial()

    def check_star_virial(self):
        if self._star_virial is None:
            raise RuntimeError("Cannot check the virial equilibrium of "
                               "the stars because there are no stars in "
                               "this model!")
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

    def set_magnetic_field_from_density(self, B0, eta=2. / 3., gaussian=True):
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

    def generate_tracer_particles(self, num_particles, r_max=None, sub_sample=1,
                                  prng=None):
        from cluster_generator.utils import parse_prng
        prng = parse_prng(prng)
        mylog.info("We will be assigning %d particles." % num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(self["radius"].d,
                                                   self["gas_mass"].d,
                                                   num_particles_sub,
                                                   r_max=r_max, prng=prng)

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(prng.uniform(low=-1., high=1., size=num_particles))
        phi = 2. * np.pi * prng.uniform(size=num_particles)

        fields = OrderedDict()

        fields["tracer", "particle_position"] = unyt_array(
            [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi),
             radius * np.cos(theta)], "kpc").T

        fields["tracer", "particle_velocity"] = unyt_array(
            np.zeros(fields["tracer", "particle_position"].shape), "kpc/Myr"
        )

        fields["tracer", "particle_mass"] = unyt_array(
            np.zeros(num_particles), "Msun"
        )

        return ClusterParticles("tracer", fields)

    def generate_gas_particles(self, num_particles, r_max=None, sub_sample=1,
                               compute_potential=False, prng=None):
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
        #  Setup
        # ------------------------------------------------------------------------------------------------------------ #
        prng = parse_prng(prng)
        mylog.info("We will be assigning %d particles." % num_particles)

        mylog.info("\tComputing particle positions...")

        # Determining the number of particles to partition into subsamples.
        num_particles_sub = num_particles // sub_sample

        #  Sampling
        # ------------------------------------------------------------------------------------------------------------ #
        # ** --------------- Radii --------------------** #
        # Inverse distribution sampling to get radii for particles and measure of total mass.
        radius_sub, mtot = generate_particle_radii(self["radius"].d,
                                                   self["gas_mass"].d,
                                                   num_particles_sub,
                                                   r_max=r_max, prng=prng)
        # duplicating the radii sampled
        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        # ** ------------------ Angular Position --------------- ** #
        theta = np.arccos(prng.uniform(low=-1., high=1., size=num_particles))
        phi = 2. * np.pi * prng.uniform(size=num_particles)

        #  Building the fields
        # ------------------------------------------------------------------------------------------------------------ #
        fields = OrderedDict()

        fields["gas", "particle_position"] = unyt_array(
            [radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi),
             radius * np.cos(theta)], "kpc").T

        mylog.info("Computing particle thermal energies, densities, and masses.")

        # --- Fetching energy --- #
        # ! Using equipartition theorem and ideal gas law.
        e_arr = 1.5 * self.fields["pressure"] / self.fields["density"]
        get_energy = InterpolatedUnivariateSpline(self.fields["radius"], e_arr)

        if sub_sample > 1:
            energy = np.tile(get_energy(radius_sub), sub_sample)[:num_particles]
        else:
            energy = get_energy(radius)

        fields["gas", "thermal_energy"] = unyt_array(energy, "kpc**2/Myr**2")

        # --- Fetching the particle mass --- #
        fields["gas", "particle_mass"] = unyt_array(
            [mtot / num_particles] * num_particles, "Msun")

        get_density = InterpolatedUnivariateSpline(self.fields["radius"],
                                                   self.fields["density"])

        if sub_sample > 1:
            density = np.tile(get_density(radius_sub),
                              sub_sample)[:num_particles]
        else:
            density = get_density(radius)

        fields["gas", "density"] = unyt_array(density, "Msun/kpc**3")

        # --- Setting particle velocities --- #
        # ! These are set to zero because they already have a thermally implied velocity
        mylog.info("Set particle velocities to zero.")

        fields["gas", "particle_velocity"] = unyt_array(
            np.zeros((num_particles, 3)), "kpc/Myr")

        # --- Setting the potential if required --- #
        if compute_potential:
            energy_spline = InterpolatedUnivariateSpline(
                self["radius"].d, -self["gravitational_potential"])
            phi = -energy_spline(radius_sub)
            if sub_sample > 1:
                phi = np.tile(phi, sub_sample)
            fields["gas", "particle_potential"] = unyt_array(
                phi, "kpc**2/Myr**2")

        return ClusterParticles("gas", fields)

    def generate_dm_particles(self, num_particles, r_max=None, sub_sample=1,
                              compute_potential=False, prng=None):
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
            num_particles, r_max=r_max, sub_sample=sub_sample,
            compute_potential=compute_potential, prng=prng)

    def generate_star_particles(self, num_particles, r_max=None, sub_sample=1,
                                compute_potential=False, prng=None):
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
            num_particles, r_max=r_max, sub_sample=sub_sample,
            compute_potential=compute_potential, prng=prng)

    def plot(self, field, rmin=None, rmax=None, fig=None, ax=None, **kwargs):
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
        if np.amax(self[field].v) > 0:
            ax.loglog(self["radius"], self[field], **kwargs)
        else:
            ax.loglog(self["radius"],-self[field],**kwargs)
        ax.set_xlim([rmin, rmax])
        ax.set_xlabel("Radius (kpc)")
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        return fig, ax

    def mass_in_radius(self, radius):
        masses = {}
        r = self.fields["radius"].to_value("kpc")
        for mtype in ["total", "gas", "dark_matter", "stellar"]:
            if f"{mtype}_mass" in self.fields:
                masses[mtype] = self.fields[f"{mtype}_mass"][r < radius][-1]
        return masses


# This is only for backwards-compatibility
class HydrostaticEquilibrium(ClusterModel):
    pass

# -------------------------------------------------------------------------------------------------------------------- #
# Functions ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def _compute_total_mass(fields,gravity,attrs):
    """
    Computes the total mass from the provided fields for the specific form of gravity.
    Parameters
    ----------
    fields: dict
        The fields from which to calculate.
    gravity: str
        The gravity type to use.

    Returns
    -------
    unyt_array
    """
    _required_fields = ["gravitational_field","radius"]

    #  Sanity checks
    # ----------------------------------------------------------------------------------------------------------------- #
    # - Checking fields - #
    if any(_f not in fields for _f in _required_fields):
        raise ValueError(f"Fields must include the required_fields: {_required_fields}")

    # - Checking on the necessary attributes - #
    if attrs is None:
        attrs = {}

    if gravity != "Newtonian":
        if "interp_function" not in attrs:
            mylog.warning(f"Gravity {gravity} requires kwarg `interp_function`. Setting to default...")
            attrs["interp_function"] = _default_interpolation_function
        if "a_0" not in attrs:
            mylog.warning(f"Gravity {gravity} requires kwarg `a_0`. Setting to default...")
            attrs["a_0"] = _default_a_0

    #  Managing the cases
    # ----------------------------------------------------------------------------------------------------------------- #
    if gravity == "Newtonian":
        # We can just use the basic equation:
        _output = -fields["radius"]**2 * fields["gravitational_field"] / G

    elif gravity == "AQUAL":
        # Computing the mass from AQUAL poisson equation
        _output = (-fields["radius"]**2*fields["gravitational_field"]/G)*attrs["interp_function"](np.abs(fields["gravitational_field"].to("kpc/Myr**2").d)/attrs["a_0"].to("kpc/Myr**2").d)

    elif gravity == "QUMOND":
        # Compute the mass from QUMOND poisson equation

        # - Creating a function equivalent of the gravitational field - #
        _gravitational_field_spline = InterpolatedUnivariateSpline(fields["radius"].d,fields["gravitational_field"].to("kpc/Myr**2").d/attrs["a_0"].to("kpc/Myr**2").d)

        _fsolve_function = lambda x: attrs["interp_function"](x)*x + _gravitational_field_spline(fields["radius"].d)

        # - Computing the guess - #
        _x_guess = np.sqrt(-_gravitational_field_spline(fields["radius"].d))

        # - solving - #
        _x = fsolve(_fsolve_function,_x_guess)

        _output = (attrs["a_0"]*fields["radius"]**2/G)*unyt_array(_x,"kpc/Myr**2")

    else:
        raise ValueError(f"{gravity} is not a valid gravity theory.")
    return _output

if __name__ == '__main__':
    from cluster_generator.tests.utils import generate_mdr_potential
    m,d,r = generate_mdr_potential()
    model = ClusterModel({"total_mass":m,"radius":r,"total_density":d},gravity="QUMOND")
    print(model.pot.units)
    print(model)
