"""
Tools for working with gravitational potentials and alternative gravity theories.

Available Gravity Theories
==========================

+------------------+---------------------------------------+
| Name             | Implementations                       |
+==================+=======================================+
| Newtonian        | ``Newtonian``: standard gravity       |
+------------------+---------------------------------------+
| MOND (classical) | - ``QUMOND``: Quasi-linear MOND [1]_  |
|                  | - ``AQUAL``: Aquadratic MOND [2]_     |
+------------------+---------------------------------------+

Notes
-----

Theory
======

References
----------
.. [1] Monthly Notices of the Royal Astronomical Society, Volume 403, Issue 2, pp. 886-895.
.. [2] Astrophysical Journal, Part 1 (ISSN 0004-637X), vol. 286, Nov. 1, 1984, p. 7-14. Research supported by the MINERVA Foundation.
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
_default_interpolation_function = lambda x: x / (1 + (x ** 20)) ** (1 / 20)
_default_a_0 = unyt_array(1.2e-10, "m/s**2")
#: The factor of :math:`r_{\mathrm{max}}` at which spines are forced to begin converging.
default_adj_boundary = 1.2


# -------------------------------------------------------------------------------------------------------------------- #
# Functions ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
class Potential:
    r"""
    The ``Potential`` class is a wrapper for a set of modified 1-D poisson solvers which obtain the correct potential
    of the initialized system for use in determining other criteria of the system.
    
    .. admonition:: Feature
        
        The ``Potential`` class can facilitate the use of MOND gravities through two implementations, ``AQUAL`` and ``QUMOND``.

    Parameters
    ----------
    fields: dict
        The fields specified to the ``Potential`` object. Depending on the methodology used to determine the
        potential, certain fields must be specified. 
    gravity: str
        The type of gravity that is in use. Options are ``AQUAL``, ``QUMOND``, or ``Newtonian``.
    attrs: dict
        Attributes to pass to the ``Potential``. These may contain a variety of pieces of information contained in the
        table below:
        
        +---------------------+------------------------------------------+----------------+---------------------------------------------------+
        | Attribute Name      | Description                              | Types          | Default                                           |
        +=====================+==========================================+================+===================================================+
        | ``interp_function`` | The MOND specific interpolation function | ``callable``   | :math:`\frac{x}{x+1}`                             |
        +---------------------+------------------------------------------+----------------+---------------------------------------------------+
        | ``a_0``             | The MOND acceleration constant.          | ``unyt float`` | :math:`1.2\times 10^{-10} \mathrm{\frac{m}{s^2}}` |
        +---------------------+------------------------------------------+----------------+---------------------------------------------------+

    Notes
    -----

    Examples
    --------
    >>> from cluster_generator.tests.utils import generate_mdr_potential
    >>> import matplotlib.pyplot as plt
    >>>
    >>> m, d, r = generate_mdr_potential() #: Generating the profile from an SNFW profile
    >>>
    >>> #- Generating the potential class objects from fields -#
    >>> pot_AQUAL = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="AQUAL")
    >>> pot_NEWTONIAN = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="Newtonian")
    >>> pot_QUMOND = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="QUMOND")
    >>> #- Plotting
    >>> figure = plt.figure(figsize=(8,5))
    >>> ax = figure.add_subplot(121)
    >>> ax2 = figure.add_subplot(122)
    >>> _,_ = pot_AQUAL.plot(fig=figure,ax=ax,color="b",label="AQUAL")
    >>> _,_ = pot_NEWTONIAN.plot(fig=figure,ax=ax,color="k",label="Newtonian")
    >>> _,_ = pot_QUMOND.plot(fig=figure,ax=ax,color="r",label="QUMOND")
    >>> _ = ax.legend()
    >>> _ = ax2.loglog(pot_QUMOND["radius"].d,np.gradient(pot_QUMOND["potential"].d,pot_QUMOND["radius"].d),"r")
    >>> _ = ax2.loglog(pot_AQUAL["radius"].d,np.gradient(pot_AQUAL["potential"].d,pot_AQUAL["radius"].d),"b")
    >>> _ = ax2.loglog(pot_NEWTONIAN["radius"].d,np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d),"k")
    >>> _ = ax2.hlines(y=_default_a_0.to("kpc/Myr**2").d,xmin=np.amin(r.d),xmax=np.amax(r.d),color="c",ls=":")
    >>> _ = ax2.loglog(pot_NEWTONIAN["radius"].d,(np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d)**2)/_default_a_0.to("kpc/Myr**2").d,"r:")
    >>> _ = ax2.loglog(pot_NEWTONIAN["radius"].d,(np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d)*_default_a_0.to("kpc/Myr**2").d)**(1/2),"b:")
    >>> _ = ax2.set_xlabel("Radius [kpc]")
    >>> _ = ax2.set_ylabel(r"$\left|a\right|,\;\;\left[\mathrm{\frac{kpc}{Myr^2}}\right]$")
    >>> _ = ax.set_ylabel(r"$\left|\Phi\right|,\;\;\left[\mathrm{\frac{kpc^2}{Myr^2}}\right]$")
    >>> _ = ax2.text(0.17,6e-2,r"$\frac{a_{\mathrm{newt}^2}}{a_0}$")
    >>> _ = ax2.text(0.17,1.4e-2,r"$a_{\mathrm{newt}}$")
    >>> _ = ax2.text(0.17,4e-3,r"$\sqrt{a_0a_{\mathrm{newt}}}$")
    >>> _ = ax.set_title("Potential")
    >>> _ = ax2.set_title("Acceleration")
    >>> _ = plt.subplots_adjust(wspace=0.3)
    >>> figure.savefig("../doc/source/_images/gravity/image1.png")

    .. image:: ../_images/gravity/image1.png
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
        obj = Potential(fields, gravity, kwargs)  # initialize the object
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

        model = cls(fields, gravity, {})
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
            a_0 = self.attrs["a_0"].to("kpc/Myr**2").d

            # - Building the gamma function - #
            gamma_func = InterpolatedUnivariateSpline(rr, G.d * tmass / (a_0 * (rr ** 2)),k=2)

            # -- Redefining with an adjusted spline approach to prevent asymptotes from forming ---#
            # =====================================================================================#
            r_bound = default_adj_boundary*rr[-1]
            gamma_func__adjusted = lambda x: np.piecewise(x,
                                                          [x <= r_bound,
                                                           x > r_bound],
                                                          [gamma_func, lambda l:  gamma_func(r_bound) * (r_bound / l) ** 2])

            self["gamma"] = unyt_array(gamma_func__adjusted(rr))


            # - generating guess solution - #
            mylog.info(f"Creating AQUAL guess solution for implicit equation...")

            Gamma_func = lambda x: (1 / 2) * (
                        gamma_func__adjusted(x) + np.sqrt(gamma_func__adjusted(x) ** 2 + 4 * gamma_func__adjusted(x)))  # -> big gamma del Phi / a_0
            _guess = Gamma_func(rr)



            # - solving - #
            mylog.info(f"Optimizing implicit solution...")
            _fsolve_function = lambda x: x * self.attrs["interp_function"](x) - self["gamma"]

            _Gamma_solution = fsolve(_fsolve_function, x0=_guess)



            Gamma = InterpolatedUnivariateSpline(rr, _Gamma_solution,k=2)

            # ** Defining the adjusted Gamma solution to prevent issues with divergence of the spline. **
            #
            #
            adj_Gamma = lambda x: np.piecewise(x, [x <= r_bound, x > r_bound],
                                               [Gamma, lambda t: 4 * Gamma(r_bound) * (r_bound ** 2) / (t ** 2)])


            # - Performing the integration process - #
            gpot2 = a_0 * unyt_array(integrate_toinf(adj_Gamma, rr), "(kpc**2)/Myr**2")

            # - Finishing computation - #
            self.fields["potential"] = -gpot2
            self.fields["potential"].convert_to_units("kpc**2/Myr**2")

            self["guess_potential"] = -a_0 * unyt_array(integrate_toinf(Gamma_func, rr), "(kpc**2)/Myr**2")
            self.fields["potential"].convert_to_units("kpc**2/Myr**2")

        elif self.gravity == "QUMOND":
            # -------------------------------------------------------------------------------------------------------- #
            #  QUMOND Implementation                                                                                   #
            # -------------------------------------------------------------------------------------------------------- #
            # - Pulling arrays
            rr = self["radius"].to("kpc").d
            tmass = self["total_mass"].to("Msun").d
            a_0 = self.attrs["a_0"].to("kpc/Myr**2").d

            ## -- Preparing for Execution -- ##
            mylog.info(f"Integrating gravitational potential profile. gravity={self.gravity}.")

            gamma_func = InterpolatedUnivariateSpline(rr, (G.d * tmass) / (a_0 * (rr ** 2)))

            self["gamma"] = unyt_array(gamma_func(rr))

            gpot_profile = lambda r: - gamma_func(r) * a_0 * self.attrs["interp_function"](gamma_func(r))

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cluster_generator.tests.utils import generate_mdr_potential

    m, d, r = generate_mdr_potential()
    pot_AQUAL = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="AQUAL")
    pot_NEWTONIAN = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="Newtonian")

    figure = plt.figure()
    ax = figure.add_subplot(121)
    ax2 = figure.add_subplot(122)
    pot_AQUAL.plot(fig=figure,ax=ax)
    pot_NEWTONIAN.plot(fig=figure,ax=ax)
    ax2.loglog(pot_AQUAL["radius"].d,np.gradient(pot_AQUAL["potential"].d,pot_AQUAL["radius"].d))
    ax2.loglog(pot_NEWTONIAN["radius"].d,np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d))
    plt.show()
