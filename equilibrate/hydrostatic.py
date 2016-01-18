import numpy as np
from collections import OrderedDict
from equilibrate.utils import \
    logspace, \
    InterpolateSplineWithUnits, \
    InterpolatedUnivariateSpline, \
    YTArray, pc, \
    setup_units, \
    mylog, \
    integrate_mass, \
    integrate_toinf
from equilibrate.equilibrium_model import EquilibriumModel

modes = {"dens_temp":("density","temperature"),
         "dens_tden":("density","total_density"),
         "dens_grav":("density","gravitational_field"),
         "tden_only":("total_density",)}

class RequiredProfilesError(Exception):
    def __init__(self, mode):
        self.mode = mode

    def __str__(self):
        ret = "Not all of the required profiles for mode \"%s\" have been set!\n" % self.mode
        ret += "The following profiles are needed: %s" % modes[self.mode]
        return ret

class HydrostaticEquilibrium(EquilibriumModel):

    _type_name = "hydrostatic"
    _xfield = None

    default_fields = ["density","temperature","pressure","total_density",
                      "gravitational_potential","gravitational_field",
                      "total_mass","thermal_energy"]

    @classmethod
    def from_scratch(cls, mode, xmin, xmax, profiles, input_units=None,
                     parameters=None, num_points=1000, geometry="spherical"):
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
                "tden_only": Generate profiles of gravitational potential
                and acceleration assuming an initial total density profile.
        xmin : YTQuantity
            The minimum radius or height for the profiles.
        xmax : YTQuantity
            The maximum radius or height for the profiles.
        profiles : dict of functions
            A dictionary of callable functions of radius or height which return
            YTArrays of quantities such as density, total density, and so on.
        input_units : dict of strings
            The default units for the different basic dimensional quantities,
            such as length, time, etc.
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

        if mode == "dens_tden" or mode == "tden_only":
            if "density" in profiles and "total_density" not in profiles:
                mylog.warning("Mode \"%s\" was specified, but only with "
                              "a density profile. Assuming density == total_density.")
                profiles["total_density"] = profiles["density"]

        for p in modes[mode]:
            if p not in profiles:
                raise RequiredProfilesError(mode)

        if mode in ["dens_tden","tden_only"] and geometry != "spherical":
            raise NotImplemented("Constructing a HydrostaticEquilibrium from gas density and/or "
                                 "total density profiles is only allowed in spherical geometry!")

        extra_fields = [field for field in profiles if field not in cls.default_fields]

        units = setup_units(input_units)

        if geometry == "cartesian":
            x_field = "height"
        elif geometry == "spherical":
            x_field = "radius"

        fields = OrderedDict()

        xx = logspace(xmin, xmax, num_points)
        fields[x_field] = xx

        mp = pc.mp.in_units(units["mass"])
        G = pc.G.in_units(units["G"])
        kB = pc.kboltz.in_units(units["kB"])

        if parameters is None:
            parameters = {}

        if "gamma" in parameters:
            gamma = parameters["gamma"]
        else:
            gamma = 5./3.
            parameters["gamma"] = gamma

        if "mu" in parameters:
            muinv = 1./parameters["mu"]
        else:
            muinv = 0.75/0.5 + 0.25/(4./3.)
            parameters["mu"] = 1.0/muinv

        parameters["geometry"] = geometry

        if mode != "tden_only":
            fields["density"] = profiles["density"](xx).in_units(units["density"])

        if mode == "dens_temp":

            mylog.info("Computing the profiles from density and temperature.")

            fields["temperature"] = profiles["temperature"](xx).in_units(units["temperature"])
            fields["pressure"] = kB*fields["density"]*fields["temperature"]
            fields["pressure"] *= muinv/mp

            pressure_spline = InterpolateSplineWithUnits(xx, fields["pressure"])
            dPdx = pressure_spline(xx, 1)
            fields["gravitational_field"] = dPdx/fields["density"]

        else:

            if mode == "dens_tden" or mode == "tden_only":
                mylog.info("Computing the profiles from density and total density.")
                tdens = profiles["total_density"](xx).in_units(units["density"])
                fields["total_density"] = tdens
                mylog.info("Integrating total mass profile.")
                fields["total_mass"] = YTArray(integrate_mass(profiles["total_density"].unitless(), xx.d),
                                               units["mass"])
                fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
            elif mode == "dens_grav":
                mylog.info("Computing the profiles from density and gravitational acceleration.")
                grav = profiles["gravitational_field"](xx).in_units(units["acceleration"])
                fields["gravitational_field"] = grav

            if mode != "tden_only":
                g = InterpolatedUnivariateSpline(xx.d, fields["gravitational_field"].d)
                dens = profiles["density"].unitless()
                dPdr_int = lambda r: dens(r)*g(r)
                mylog.info("Integrating pressure profile.")
                fields["pressure"] = -YTArray(integrate_toinf(dPdr_int, xx.d), units["pressure"])
                fields["temperature"] = fields["pressure"]/fields["density"]
                fields["temperature"] *= mp/muinv

        if mode != "tden_only":
            fields["thermal_energy"] = fields["pressure"]/(gamma-1.)/fields["density"]

        if geometry == "spherical":
            if "total_mass" not in fields:
                fields["total_mass"] = fields["radius"]**2*fields["gravitational_field"]/G
                total_mass_spline = InterpolateSplineWithUnits(xx, fields["total_mass"])
                dMdr = total_mass_spline(xx, 1)
                fields["total_density"] = dMdr/(4.*np.pi*xx**2)
            mylog.info("Integrating gravitational potential profile.")
            tden = profiles["total_density"].unitless()
            gpot_profile = lambda r: tden(r)*r
            gpot = YTArray(4.*np.pi*integrate_toinf(gpot_profile, xx.d), 
                           units["density"]*units["length"]**2)
            fields["gravitational_potential"] = -G*(fields["total_mass"]/fields["radius"] + gpot)
            if mode != "tden_only":
                mylog.info("Integrating gas mass profile.")
                fields["gas_mass"] = YTArray(integrate_mass(profiles["density"].unitless(), xx.d),
                                             units["mass"])

        for field in extra_fields:
            fields[field] = profiles[field](xx)

        return cls(num_points, fields, parameters, units)

    def compute_dark_matter_profiles(self):
        r"""
        If the total density profile and the gas density profile
        are different, assume the rest is dark matter. Compute
        the dark matter density and mass profiles and store them.
        """
        mdm = self["total_mass"]-self["gas_mass"]
        ddm = self["total_density"]-self["density"]
        mdm[ddm < 0.0][:] = mdm.max()
        ddm[ddm < 0.0][:] = 0.0
        self.set_field("dark_matter_density", ddm)
        self.set_field("dark_matter_mass", mdm)

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
        xx = self.fields[self.x_field]
        pressure_spline = InterpolateSplineWithUnits(xx, self.fields["pressure"])
        dPdx = pressure_spline(xx,1)
        rhog = self.fields["density"]*self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info("The maximum relative deviation of this profile from "
                   "hydrostatic equilibrium is %g" % np.abs(chk).max())
        return chk