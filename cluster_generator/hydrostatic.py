import numpy as np
from collections import OrderedDict
from yt import YTArray, YTQuantity, mylog
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import \
    integrate, \
    integrate_mass, \
    mp, G, generate_particle_radii
from cluster_generator.cluster_model import ClusterModel, \
    ClusterParticles

gamma = 5./3.

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

    default_fields = ["density","temperature","pressure","total_density",
                      "gravitational_potential","gravitational_field",
                      "total_mass","gas_mass","dark_matter_mass",
                      "dark_matter_density"]

    @classmethod
    def from_scratch(cls, mode, rmin, rmax, profiles, num_points=1000,
                     P_amb=0.0, mu=None):
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
        rmin : float
            The minimum radius for the profiles, assumed to be in kpc.
        rmax : float
            The maximum radius for the profiles, assumed to be in kpc.
        profiles : dict of functions
            A dictionary of callable functions of radius or height which return
            quantities such as density, total density, and so on. The functions 
            are not unit-aware for speed purposes, but they assume that the base
            units are:
                "length": "kpc"
                "time": "Myr"
                "mass": "Msun"
                "temperature": "keV"
        num_points : integer
            The number of points at which to evaluate the profile.
        P_amb : float, optional
            The ambient pressure in units of erg/cm**3, used as a boundary
            condition on the pressure integral. Default: 0.0.
        mu : float, optional
            The mean molecular weight. If not specified, it will be calculated
            assuming a fully ionized gas with primordial abundances.
        """
        if mu is None:
            muinv = 0.76/0.5 + 0.24/(4./3.)
            mu = 1.0/muinv
        else:
            muinv = 1.0/mu

        for k, p in profiles.items():
            if hasattr(p, "unitless"):
                profiles[k] = p.unitless()

        if not isinstance(P_amb, YTQuantity):
            P_amb = YTQuantity(P_amb, "erg/cm**3")
        P_amb.convert_to_units("Msun/(Myr**2*kpc)")

        for p in modes[mode]:
            if p not in profiles:
                raise RequiredProfilesError(mode)

        extra_fields = [field for field in profiles if field not in cls.default_fields]

        fields = OrderedDict()

        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields["radius"] = YTArray(rr, "kpc")

        if mode == "dm_only":
            fields["density"] = YTArray(np.zeros(num_points), "Msun/kpc**3")
            fields["pressure"] = YTArray(np.zeros(num_points), "Msun/kpc/Myr**2")
            fields["temperature"] = YTArray(np.zeros(num_points), "keV")
        else:
            fields["density"] = YTArray(profiles["density"](rr), "Msun/kpc**3")

        if mode == "dens_temp":

            mylog.info("Computing the profiles from density and temperature.")

            fields["temperature"] = YTArray(profiles["temperature"](rr), "keV")
            fields["pressure"] = fields["density"]*fields["temperature"]
            fields["pressure"] *= muinv/mp
            fields["pressure"].convert_to_units("Msun/(Myr**2*kpc)")

            pressure_spline = InterpolatedUnivariateSpline(rr, fields["pressure"].v)
            dPdx = YTArray(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
            fields["gravitational_field"] = dPdx/fields["density"]
            fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        else:

            if mode == "dens_tden" or mode == "dm_only":
                mylog.info("Computing the profiles from density and total density.")
                fields["total_density"] = YTArray(profiles["total_density"](rr), "Msun/kpc**3")
                mylog.info("Integrating total mass profile.")
                fields["total_mass"] = YTArray(integrate_mass(profiles["total_density"], rr), "Msun")
                fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
                fields["gravitational_field"].convert_to_units("kpc/Myr**2")
            elif mode == "dens_grav":
                mylog.info("Computing the profiles from density and gravitational acceleration.")
                fields["gravitational_field"] = YTArray(profiles["gravitational_field"](rr), "kpc/Myr**2")

            if mode != "dm_only":
                g = fields["gravitational_field"].in_units("kpc/Myr**2").v
                g_r = InterpolatedUnivariateSpline(rr, g)
                dPdr_int = lambda r: profiles["density"](r)*g_r(r)
                mylog.info("Integrating pressure profile.")
                fields["pressure"] = -YTArray(integrate(dPdr_int, rr), "Msun/kpc/Myr**2")
                fields["temperature"] = fields["pressure"]*mp/fields["density"]/muinv
                fields["temperature"].convert_to_units("keV")

        if "total_mass" not in fields:
            fields["total_mass"] = -fields["radius"]**2*fields["gravitational_field"]/G
        if "total_density" not in fields:
            total_mass_spline = InterpolatedUnivariateSpline(rr, fields["total_mass"].v)
            dMdr = YTArray(total_mass_spline(rr, 1), "Msun/kpc")
            fields["total_density"] = dMdr/(4.*np.pi*fields["radius"]**2)
        mylog.info("Integrating gravitational potential profile.")
        if "total_density" in profiles:
            tdens_func = profiles["total_density"]
        else:
            tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)
        gpot_profile = lambda r: tdens_func(r)*r
        gpot = YTArray(4.*np.pi*integrate(gpot_profile, rr), "Msun/kpc")
        fields["gravitational_potential"] = -G*(fields["total_mass"]/fields["radius"] + gpot)
        fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")
        if mode != "dm_only":
            mylog.info("Integrating gas mass profile.")
            fields["gas_mass"] = YTArray(integrate_mass(profiles["density"], rr), "Msun")

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
            fields[field] = profiles[field](rr)

        return cls(num_points, fields, parameters={"mu": mu})

    def check_model(self):
        r"""
        Determine the deviation of the model from hydrostatic equilibrium. Returns
        an array containing the relative deviation at each radius or height.
        """
        rr = self.fields["radius"].v
        pressure_spline = InterpolatedUnivariateSpline(rr, self.fields["pressure"].v)
        dPdx = YTArray(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        rhog = self.fields["density"]*self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info("The maximum relative deviation of this profile from "
                   "hydrostatic equilibrium is %g" % np.abs(chk).max())
        return chk

    def generate_particles(self, num_particles, r_max=None):
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
        """
        mu = self.parameters["mu"]

        mylog.info("We will be assigning %d particles." % num_particles)
        mylog.info("Compute particle positions.")

        radius, mtot = generate_particle_radii(self["radius"].d,
                                               self["gas_mass"].d,
                                               num_particles, r_max=r_max)

        theta = np.arccos(np.random.uniform(low=-1., high=1., size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields = OrderedDict()

        fields["gas", "particle_position"] = YTArray([radius*np.sin(theta)*np.cos(phi),
                                                      radius*np.sin(theta)*np.sin(phi),
                                                      radius*np.cos(theta)], "kpc").T

        mylog.info("Compute particle thermal energies, densities, and masses.")

        e_arr = 1.5*self.fields["pressure"]/self.fields["density"]
        get_energy = InterpolatedUnivariateSpline(self.fields["radius"], e_arr)
        e_int = get_energy(radius)

        fields["gas", "particle_thermal_energy"] = YTArray(e_int, "kpc**2/Myr**2")
        fields["gas", "particle_temperature"] = fields["gas", "particle_thermal_energy"]*mu*mp/1.5
        fields["gas", "particle_temperature"].convert_to_units("keV")

        fields["gas", "particle_mass"] = YTArray([mtot/num_particles]*num_particles, "Msun")

        get_density = InterpolatedUnivariateSpline(self.fields["radius"], self.fields["density"])

        fields["gas", "particle_density"] = YTArray(get_density(radius), "Msun/kpc**3")

        mylog.info("Set particle velocities to zero.")

        fields["gas", "particle_velocity"] = YTArray(np.zeros((num_particles, 3)), "kpc/Myr")

        return ClusterParticles("gas", fields)
