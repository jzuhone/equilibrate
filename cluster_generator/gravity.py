r"""
Gravity
=======
The :py:mod:`gravity` module contains the :py:class:`gravity.Potential` class, which essentially acts as a wrapper for
solving the Poisson equation in each of the available gravitational theories. Potential objects contain ``fields`` and
can be written to / read from binary files.

.. attention::

    It is easy to confuse the :py:class:`gravity.Potential` class and the :py:class:`model.ClusterModel` class. The :py:class:`gravity.Potential` class is inherited by
    the :py:class:`model.ClusterModel` class and holds all of the core data (``fields``, ``gravity`` , ``attributes`` , etc.); however, :py:class:`gravity.Potential` objects
    **do not** have the capacity to generate particles, be virialized, or be checked against hydrostatic equilibrium.

    In general, it is only rarely useful to initialize a ``Potential`` instance instead of simply using the ``ClusterModel`` instance.
"""
import os
from collections import OrderedDict
import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import fsolve
from unyt import unyt_array
from halo import Halo
from cluster_generator.utils import \
    integrate, mylog, G, integrate_toinf, cg_params,truncate_spline,log_string
import sys

# -------------------------------------------------------------------------------------------------------------------- #
# Functions ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
class Potential:
    r"""
    The :py:class:`gravity.Potential` class underlies all of the galaxy cluster models used in the ``cluster_generator`` library. This underlying
    structure is used to contain / manage ``fields`` (data profiles) and compute potentials from them using a variety of gravities.

    Parameters
    ----------
    fields: dict of {str: unyt_array}
        The fields parameter should include ``unyt_array`` objects for each of the fields the user wants to implement. Any field name is valid; however,
        standard names should be used for fields that are necessary for generation / manipulation of the data.
    gravity: str
        The gravity theory to use. The list of available gravitational theories can be found at :ref:`gravity`.
    **kwargs
        Additional parameters may be passed to the :py:class:`gravity.Potential` instance through the use of ``**kwargs``. Generally these are not necessary, or are
        automatically generated as necessary behind the scenes; however, some special kwargs do exist of which the user should be aware:

        +---------------------------------+-----------------------------------------+------------------------------------+----------------------------------------------------------------------------------------------------+
        | kwarg name                      | kwarg                                   | Description                        | Has Default?                                                                                       |
        +=================================+=========================================+====================================+====================================================================================================+
        | Interpolation Function          | ``interp_function``                     | The MONDian interpolation function |                                                                                                    |
        |                                 |                                         |                                    |AQUAL: :math:`\mu(x)=x/(1+x^\alpha)^{1/\alpha}`                                                     |
        |                                 |                                         |                                    |QUMOND: :math:`\nu(x)=\left[\frac{1}{2}\left(1+\sqrt{1+\frac{4}{x^\alpha}}\right)\right]^{1/\alpha}`|
        +---------------------------------+-----------------------------------------+------------------------------------+----------------------------------------------------------------------------------------------------+
        | :math:`a_0`                     | ``a_0``                                 | The MONDian :math:`a_0` constant   | ``True``: :math:`a_0 = 1.2 \times 10^{-10} \mathrm{m\;s^{-2}}`                                     |
        +---------------------------------+-----------------------------------------+------------------------------------+----------------------------------------------------------------------------------------------------+
    """
    _keep_units = ["entropy", "electron_number_density",
                   "magnetic_field_strength"]
    _methods = {
        "mdr": ["radius", "total_mass", "total_density"]
    }

    def __init__(self, fields, gravity, **kwargs):
        # - Initializing base attributes - #
        #: The ``fields`` associated with the ``Potential`` object.
        self.fields = fields

        #: The gravity type of the ``Potential`` instance.
        self.gravity = gravity

        # - Derived attributes -#
        #: The number of elements in each ``field`` array.
        self.num_elements = len(self.fields["radius"])

        if "gravitational_potential" not in self.fields:
            self.fields["gravitational_potential"] = None

        #: Additional attributes associated with the object. Derived from ``**kwargs``.
        self.attrs = kwargs

        #  Managing forced attributes for MONDian gravity theories
        # ------------------------------------------------------------------------------------------------------------ #
        if gravity != "Newtonian":
            if "interp_function" not in self.attrs:
                mylog.warning(f"Gravity {self.gravity} requires kwarg `interp_function`. Setting to default...")
                self.attrs["interp_function"] = cg_params["mond", f"{gravity}_interp"]
            if "a_0" not in self.attrs:
                mylog.warning(f"Gravity {self.gravity} requires kwarg `a_0`. Setting to default...")
                self.attrs["a_0"] = cg_params["mond", "a_0"]

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
        """The gravitational potential of the system. Equivalent to ``self.fields["gravitational_potential"]`` or ``self["gravitational_potential"]``.
            If the ``self["gravitational_potential"]`` field is ``None``, calling ``self.pot`` will automatically attempt to generate a potential.
            """
        if self.fields["gravitational_potential"] is not None:
            return self.fields["gravitational_potential"]
        else:
            try:
                self.potential()
                return self.fields["gravitational_potential"]
            except ValueError as exception:
                raise ValueError(
                    f"Failed to compute a potential from available fields. Message = {exception.__repr__()}")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Class Methods ================================================================================================== #
    # ---------------------------------------------------------------------------------------------------------------- # 
    @classmethod
    def from_fields(cls, fields, gravity="Newtonian", **kwargs):
        """
        Initializes a :py:class:`gravity.Potential` object from its constituent fields.

        .. attention::

            The :py:meth:`gravity.Potential.from_fields` is entirely equivalent to simply generating an instance from the
            ``.__init__()``  method; however, :py:meth:`gravity.Potential.from_fields` generates the potential automatically, without
            requiring the user to call ``_ = self.pot`` to generate the potential.

        Parameters
        ----------
        fields: dict
            The fields of data from which to generate the ``Potential`` object.
        gravity: str, optional, default="Newtonian"
            The type of gravity to use in computations. Options are ``Newtonian``,``AQUAL``, or ``QUMOND``.
        **kwargs:
            Additional key-word arguments to be parsed to ``self.attrs`` after instantiation.
        Returns
        -------
        Potential
            The resultant potential.
        """
        #  Logging
        # ----------------------------------------------------------------------------------------------------------------- #
        mylog.info(f"Computing gravitational potential from fields. gravity={gravity}.")

        #  Sanity Check
        # ----------------------------------------------------------------------------------------------------------------- #
        if all(any(field not in fields for field in req_fields) for req_fields in list(cls._methods.values())):
            raise ValueError(f"The fields {list(fields.keys())} are not sufficient for a potential computation.")
        else:
            _used_method = \
            [key for key, value in cls._methods.items() if all(req_field in fields for req_field in value)][
                0]
            mylog.info(f"Computation of potential is using {_used_method} for computation.")

        #  Computing Potential
        # ----------------------------------------------------------------------------------------------------------------- #
        obj = Potential(fields, gravity, **kwargs)  # initialize the object
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

        Notes
        -----
        .. warning::

            The :py:meth:`gravity.Potential.from_h5_file` method will attempt to load both the ``HDF5`` file ``filename``, but will
            also look for a ``filename.pkl`` file, containing the serialized version of the ``self.attrs`` dictionary. If it fails to find the
            attribute file, the data will still be loaded, but attributes may be lost.

        """
        import dill as pickle
        # - Grabbing base data -#
        with h5py.File(filename, "r") as f:
            fnames = list(f['fields'].keys())

        # - reading attributes - #
        atr_path = f"{filename}.pkl"

        if os.path.exists(atr_path):
            with open(atr_path, "rb") as atf:
                attrs = pickle.load(atf)
        else:
            mylog.warning(f"[[HDF5]] Failed to locate the pkl attribute file for {filename}.")
            attrs = {"gravity": "Newtonian"}

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

        gravity = attrs["gravity"]
        del attrs["gravity"]
        model = cls(fields, gravity, **attrs)
        return model

    # ---------------------------------------------------------------------------------------------------------------- #
    # Methods ======================================================================================================== #
    # ---------------------------------------------------------------------------------------------------------------- # 

    def potential(self, method=None):
        """
        Computes the potential of the ``Potential`` object using any of a number of methods.

        Parameters
        ----------
        method: str, default=None
            The preferred method for computing the potential. If ``None``, then the easiest available option fitting
            the available fields will be used.

        Returns
        -------
        None

        Notes
        -----

        Available Methods
        +++++++++++++++++

        .. admonition:: TODO

            Write this documentation

        """
        #  Logging
        # ----------------------------------------------------------------------------------------------------------------- #
        mylog.info(f"Computing gravitational potential of {self.__repr__()}. gravity={self.gravity}.")

        #  Sanity Check
        # ----------------------------------------------------------------------------------------------------------------- #
        if all(any(field not in self.fields for field in req_fields) for req_fields in list(self._methods.values())):
            raise ValueError(
                f"The self.fields {list(self.fields.keys())} are not sufficient for a potential computation.")
        else:
            if method:
                _used_method = method
            else:
                _used_method = \
                    [key for key, value in self._methods.items() if
                     all(req_field in self.fields for req_field in value)][
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

            mylog.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={self.gravity}.")
            tdens_func = InterpolatedUnivariateSpline(rr, self.fields["total_density"].d)

            # - Computing - #
            gpot_profile = lambda r: tdens_func(r) * r

            gpot1 = self.fields["total_mass"] / self.fields["radius"]
            gpot2 = unyt_array(4. * np.pi * integrate(gpot_profile, rr), "Msun/kpc")

            # - Finishing computation - #

            self.fields["gravitational_potential"] = -G * (gpot1 + gpot2)
            self.fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

        elif self.gravity == "AQUAL":
            # -------------------------------------------------------------------------------------------------------- #
            #  AQUAL Implementation                                                                                    #
            # -------------------------------------------------------------------------------------------------------- #
            mylog.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={self.gravity}.")

            # - pulling arrays
            rr = self["radius"].d
            tmass = self["total_mass"].d
            a_0 = self.attrs["a_0"].to("kpc/Myr**2").d

            # - Building the gamma function - #
            gamma_func = InterpolatedUnivariateSpline(rr, G.d * tmass / (a_0 * (rr ** 2)), k=2)

            # -- Redefining with an adjusted spline approach to prevent asymptotes from forming ---#
            # =====================================================================================#
            r_bound = cg_params["util", "adj_factor"] * rr[-1]
            gamma_func__adjusted = lambda x: np.piecewise(x,
                                                          [x <= r_bound,
                                                           x > r_bound],
                                                          [gamma_func,
                                                           lambda l: gamma_func(r_bound) * (r_bound / l) ** 2])

            self["gamma"] = unyt_array(gamma_func__adjusted(rr))

            # - generating guess solution - #
            mylog.debug(f"[[Potential]] Creating AQUAL guess solution for implicit equation...")

            Gamma_func = lambda x: (1 / 2) * (
                    gamma_func__adjusted(x) + np.sqrt(
                gamma_func__adjusted(x) ** 2 + 4 * gamma_func__adjusted(x)))  # -> big gamma del Phi / a_0
            _guess = Gamma_func(rr)

            # - solving - #
            mylog.debug(f"[[Potential]] Optimizing implicit solution...")
            _fsolve_function = lambda x: x * self.attrs["interp_function"](x) - self["gamma"]

            _Gamma_solution = fsolve(_fsolve_function, x0=_guess)

            Gamma = InterpolatedUnivariateSpline(rr, _Gamma_solution, k=2)

            # ** Defining the adjusted Gamma solution to prevent issues with divergence of the spline. **
            #
            #
            adj_Gamma = truncate_spline(Gamma,0.95*rr[-1],7)

            # - Performing the integration process - #
            with Halo(text=log_string("Integrating Gamma profile..."),spinner="dots",stream=sys.stderr,animation="marquee") as h:
                gpot2 = a_0 * unyt_array(integrate_toinf(adj_Gamma, rr), "(kpc**2)/Myr**2")
                h.succeed(text=log_string("Integrated Gamma profile."))
            # - Finishing computation - #
            self.fields["gravitational_potential"] = -gpot2
            self.fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")


        elif self.gravity == "QUMOND":
            # -------------------------------------------------------------------------------------------------------- #
            #  QUMOND Implementation                                                                                   #
            # ------------------------#                                                                                #
            # In the far field, the acceleration in QUMOND goes as 1/r, which is not integrable. As such, our scheme   #
            # is to set the gauge at 2*rmax and then integrate out to it. Because we only need the potential out to rr #
            # its fine to not have an entirely integrated curve.                                                       #
            # -------------------------------------------------------------------------------------------------------- #
            # - Pulling arrays
            rr = self["radius"].to("kpc").d
            rr_max = 2 * rr[-1]
            tmass = self["total_mass"].to("Msun").d
            a_0 = self.attrs["a_0"].to("kpc/Myr**2").d

            ## -- Preparing for Execution -- ##
            mylog.debug(f"[[Potential]] Integrating gravitational potential profile. gravity={self.gravity}.")

            _gamma_func = InterpolatedUnivariateSpline(rr, (G.d * tmass) / (a_0 * (rr ** 2)), k=2)
            gamma_func = lambda r: _gamma_func(r) * (1 / (1 + (r / rr_max) ** 4))
            self["gamma"] = unyt_array(gamma_func(rr))

            gpot_profile = lambda r: - gamma_func(r) * a_0 * self.attrs["interp_function"](gamma_func(r))
            # - Performing the integration process - #
            gpot2 = unyt_array(integrate(gpot_profile, rr, 2 * rr[-1]), "(kpc**2)/Myr**2")

            # - Finishing computation - #
            self.fields["gravitational_potential"] = gpot2
            self.fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")
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
        Write an instance of :py:class:`gravity.Potential` to an ``HDF5`` file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.

        Notes
        -----
        .. warning::

            The :py:meth:`gravity.Potential.write_potential_to_h5` method will attempt to write both the ``HDF5`` file ``filename``, and will
            also write a ``filename.pkl`` file, containing the serialized version of the ``self.attrs`` dictionary.

        """
        import dill as pickle
        #  Managing paths
        # ------------------------------------------------------------------------------------------------------------ #
        mylog.info(f"Writing {self} to {output_filename} in HDF5 format.")
        if os.path.exists(output_filename) and not overwrite:
            raise IOError(f"Cannot create {output_filename}. It exists and "
                          f"overwrite=False.")

        #  Writing IO
        # ------------------------------------------------------------------------------------------------------------ #
        f = h5py.File(output_filename, "w")
        f.create_dataset("num_elements", data=self.num_elements)
        f.close()

        # -- Managing the attributes -- #
        self.attrs["unit_system"] = "cgs" if in_cgs else "galactic"
        self.attrs["gravity"] = self.gravity

        with open(f"{output_filename}.pkl", "wb") as atf:
            pickle.dump(self.attrs, atf)

        # -- Writing main datasets -- #

        r_min = 0.0
        r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(self.fields["radius"].d >= r_min,
                              self.fields["radius"].d <= r_max)

        for k, v in self.fields.items():
            mylog.debug(f"[[HDF5]] Writing field {k} to file.")
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

        if self.fields["gravitational_potential"] is not None:
            mylog.debug(f"[[HDF5]] Writing field 'gravitational_potential' to file.")
            fd = self.fields["gravitational_potential"]
            fd.write_hdf5(output_filename, dataset_name="gravitational_potential", group_name="fields")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cluster_generator.radial_profiles import *
    import logging

    logger = logging.getLogger()
    logger.setLevel("DEBUG")


    def generate_mdr_potential():
        z = 0.1
        M200 = 1.5e15
        conc = 4.0
        r200 = find_overdensity_radius(M200, 200.0, z=z)
        a = r200 / conc
        M = snfw_total_mass(M200, r200, a)
        rhot = snfw_density_profile(M, a)
        m = snfw_mass_profile(M, a)

        rmin, rmax = 0.1, 2 * r200
        r = np.geomspace(rmin, rmax, 1000)
        return unyt_array(m(r), "Msun"), unyt_array(rhot(r), "Msun/kpc**3"), unyt_array(r, "kpc")


    m, d, r = generate_mdr_potential()
    p = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="AQUAL")
    p.write_potential_to_h5("test.h5")

    pot = Potential.from_h5_file("test.h5")
    print(pot.attrs)
