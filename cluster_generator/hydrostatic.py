import numpy as np
from collections import OrderedDict
from yt import YTArray, YTQuantity, mylog
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import \
    integrate, \
    integrate_mass, \
    mp, G

from cluster_generator.cluster_model import ClusterModel

gamma = 5./3.
muinv = 0.76/0.5 + 0.24/(4./3.)

modes = {"dens_temp": ("density","temperature"),
         "dens_tden": ("density","total_density"),
         "dens_grav": ("density","gravitational_field"),
         "dm_only": ("total_density",)}

class RequiredProfilesError(Exception):
    def __init__(self, mode):
        self.mode = mode

    def __str__(self):
        ret = "Not all of the required profiles for mode \"%s\" have been set!\n" % self.mode
        ret += "The following profiles are needed: %s" % modes[self.mode]
        return ret

class HydrostaticEquilibrium(ClusterModel):

    _type_name = "hydrostatic"
    _xfield = None

    default_fields = ["density","temperature","pressure","total_density",
                      "gravitational_potential","gravitational_field",
                      "total_mass","gas_mass","dark_matter_mass",
                      "dark_matter_density"]

    @classmethod
    def from_scratch(cls, mode, xmin, xmax, profiles, num_points=1000,
                     geometry="spherical", P_amb=0.0):
        r"""
        Generate a set of profiles of physical quantities based on the assumption
        of hydrostatic equilibrium. Currently assumes an ideal gas with a gamma-law
        equation of state.

        Parameters
        ----------
        mode : string
            The method to generate the profiles from an initial set. Can be
            one of the following:
                "dens_temp": Generate the profiles given a gas density and
                gas temperature profile.
                "dens_tden": Generate the profiles given a gas density and
                total density profile.
                "dens_grav": Generate the profiles given a gas density and
                gravitational acceleration profile.
                "dm_only": Generate profiles of gravitational potential
                and acceleration assuming an initial DM density profile.
        xmin : float
            The minimum radius or height for the profiles, assumed to be in kpc.
        xmax : float
            The maximum radius or height for the profiles, assumed to be in kpc.
        profiles : dict of functions
            A dictionary of callable functions of radius or height which return
            quantities such as density, total density, and so on. The functions 
            are not unit-aware for speed purposes, but they assume that the base
            units are:
                "length": "kpc"
                "time": "Myr"
                "mass": "Msun"
                "temperature": "keV"
        parameters : dict of floats
            A dictionary of parameters needed for the calculation, which include:
                "mu": The mean molecular weight of the gas. Default is to assume a
                primordial H/He gas.
                "gamma": The ratio of specific heats. Default: 5/3.
        num_points : integer
            The number of points at which to evaluate the profile.
        geometry : string
            The geometry of the model. Can be "cartesian" or "spherical", which will
            determine whether or not the profiles are of "radius" or "height".
        """

        if not isinstance(P_amb, YTQuantity):
            P_amb = YTQuantity(P_amb, "erg/cm**3")
        P_amb.convert_to_units("Msun/(Myr**2*kpc)")

        for p in modes[mode]:
            if p not in profiles:
                raise RequiredProfilesError(mode)

        if mode in ["dens_tden","dm_only"] and geometry != "spherical":
            raise NotImplemented("Constructing a HydrostaticEquilibrium from gas density and/or "
                                 "total density profiles is only allowed in spherical geometry!")

        extra_fields = [field for field in profiles if field not in cls.default_fields]

        if geometry == "cartesian":
            x_field = "height"
        elif geometry == "spherical":
            x_field = "radius"

        fields = OrderedDict()

        xx = np.logspace(np.log10(xmin), np.log10(xmax), num_points, endpoint=True)
        fields[x_field] = YTArray(xx, "kpc")

        if mode == "dm_only":
            fields["density"] = YTArray(np.zeros(num_points), "Msun/kpc**3")
            fields["pressure"] = YTArray(np.zeros(num_points), "Msun/kpc/Myr**2")
            fields["temperature"] = YTArray(np.zeros(num_points), "keV")
        else:
            fields["density"] = YTArray(profiles["density"](xx), "Msun/kpc**3")

        if mode == "dens_temp":

            mylog.info("Computing the profiles from density and temperature.")

            fields["temperature"] = YTArray(profiles["temperature"](xx), "keV")
            fields["pressure"] = fields["density"]*fields["temperature"]
            fields["pressure"] *= muinv/mp
            fields["pressure"].convert_to_units("Msun/(Myr**2*kpc)")

            pressure_spline = InterpolatedUnivariateSpline(xx, fields["pressure"].v)
            dPdx = YTArray(pressure_spline(xx, 1), "Msun/(Myr**2*kpc**2)")
            fields["gravitational_field"] = dPdx/fields["density"]
            fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        else:

            if mode == "dens_tden" or mode == "dm_only":
                mylog.info("Computing the profiles from density and total density.")
                fields["total_density"] = YTArray(profiles["total_density"](xx), "Msun/kpc**3")
                mylog.info("Integrating total mass profile.")
                fields["total_mass"] = YTArray(integrate_mass(profiles["total_density"], xx), "Msun")
                fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
                fields["gravitational_field"].convert_to_units("kpc/Myr**2")
            elif mode == "dens_grav":
                mylog.info("Computing the profiles from density and gravitational acceleration.")
                fields["gravitational_field"] = YTArray(profiles["gravitational_field"](xx), "kpc/Myr**2")

            if mode != "dm_only":
                g = fields["gravitational_field"].in_units("kpc/Myr**2").v
                g_r = InterpolatedUnivariateSpline(xx, g)
                dPdr_int = lambda r: profiles["density"](r)*g_r(r)
                mylog.info("Integrating pressure profile.")
                fields["pressure"] = -YTArray(integrate(dPdr_int, xx), "Msun/kpc/Myr**2")
                fields["temperature"] = fields["pressure"]*mp/fields["density"]/muinv
                fields["temperature"].convert_to_units("keV")

        if geometry == "spherical":
            if "total_mass" not in fields:
                fields["total_mass"] = -fields["radius"]**2*fields["gravitational_field"]/G
            if "total_density" not in fields:
                total_mass_spline = InterpolatedUnivariateSpline(xx, fields["total_mass"].v)
                dMdr = YTArray(total_mass_spline(xx, 1), "Msun/kpc")
                fields["total_density"] = dMdr/(4.*np.pi*fields["radius"]**2)
            mylog.info("Integrating gravitational potential profile.")
            if "total_density" in profiles:
                tdens_func = profiles["total_density"]
            else:
                tdens_func = InterpolatedUnivariateSpline(xx, fields["total_density"].d)
            gpot_profile = lambda r: tdens_func(r)*r
            gpot = YTArray(4.*np.pi*integrate(gpot_profile, xx), "Msun/kpc")
            fields["gravitational_potential"] = -G*(fields["total_mass"]/fields["radius"] + gpot)
            fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")
            if mode != "dm_only":
                mylog.info("Integrating gas mass profile.")
                fields["gas_mass"] = YTArray(integrate_mass(profiles["density"], xx), "Msun")

        mdm = fields["total_mass"].copy()
        ddm = fields["total_density"].copy()
        if mode != "dm_only":
            mdm -= fields["gas_mass"]
            ddm -= fields["density"]
            mdm[ddm.v < 0.0][:] = mdm.max()
            ddm[ddm.v < 0.0][:] = 0.0
        fields["dark_matter_density"] = ddm
        fields["dark_matter_mass"] = mdm

        fields['pressure'] += P_amb

        for field in extra_fields:
            fields[field] = profiles[field](xx)

        return cls(num_points, fields, geometry)

    @property
    def x_field(self):
        if self._xfield is None:
            self._xfield = list(self.keys())[0]
        return self._xfield

    def check_model(self):
        r"""
        Determine the deviation of the model from hydrostatic equilibrium. Returns
        an array containing the relative deviation at each radius or height.
        """
        xx = self.fields[self.x_field].v
        pressure_spline = InterpolatedUnivariateSpline(xx, self.fields["pressure"].v)
        dPdx = YTArray(pressure_spline(xx, 1), "Msun/(Myr**2*kpc**2)")
        rhog = self.fields["density"]*self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info("The maximum relative deviation of this profile from "
                   "hydrostatic equilibrium is %g" % np.abs(chk).max())
        return chk