import numpy as np
from collections import OrderedDict
from yt import YTArray, YTQuantity, mylog
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import \
    integrate, \
    integrate_mass, kboltz, \
    mp, G, generate_particle_radii
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cluster_particles import \
    ClusterParticles

gamma = 5./3.

modes = {"dens_temp": ("density","temperature"),
         "dens_tden": ("density","total_density"),
         "dens_grav": ("density","gravitational_field"),
         "no_gas": ("total_density",)}

muinv_default = 0.76/0.5 + 0.24/(4./3.)


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
                      "dark_matter_density","stellar_density","stellar_mass"]

    @classmethod
    def from_scratch(cls, mode, rmin, rmax, profiles, num_points=1000,
                     T_amb=1.0e5, mu=None):
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
                "no_gas": Generate profiles of gravitational potential
                and acceleration assuming an initial density profile of DM
                and/or stars.
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
        T_amb : float, optional
            The ambient temperature in units of K, used with *rho_amb* to 
            create a boundary condition on the pressure integral. Used only
            if the "dens_tden" or "dens_grav" modes are chosen. 
            Default: 1.0e5
        mu : float, optional
            The mean molecular weight. If not specified, it will be calculated
            assuming a fully ionized gas with primordial abundances.
        """
        if mu is None:
            muinv = muinv_default
            mu = 1.0/muinv
        else:
            muinv = 1.0/mu

        if not isinstance(T_amb, YTQuantity):
            T_amb = YTQuantity(T_amb, "K")

        for p in modes[mode]:
            if p not in profiles:
                raise RequiredProfilesError(mode)

        extra_fields = [field for field in profiles if field not in cls.default_fields]

        fields = OrderedDict()

        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields["radius"] = YTArray(rr, "kpc")

        if mode == "no_gas":
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

            if mode == "dens_tden" or mode == "no_gas":
                if mode == "dens_tden":
                    mylog.info("Computing the profiles from density and total density.")
                else:
                    mylog.info("Computing the profiles for dark matter only.")
                fields["total_density"] = YTArray(profiles["total_density"](rr), "Msun/kpc**3")
                mylog.info("Integrating total mass profile.")
                fields["total_mass"] = YTArray(integrate_mass(profiles["total_density"], rr), "Msun")
                fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
                fields["gravitational_field"].convert_to_units("kpc/Myr**2")
            elif mode == "dens_grav":
                mylog.info("Computing the profiles from density and gravitational acceleration.")
                fields["gravitational_field"] = YTArray(profiles["gravitational_field"](rr), "kpc/Myr**2")

            if mode != "no_gas":
                g = fields["gravitational_field"].in_units("kpc/Myr**2").v
                g_r = InterpolatedUnivariateSpline(rr, g)
                dPdr_int = lambda r: profiles["density"](r)*g_r(r)
                mylog.info("Integrating pressure profile.")
                fields["pressure"] = -YTArray(integrate(dPdr_int, rr), "Msun/kpc/Myr**2")
                P_amb = fields["density"][-1]*kboltz*T_amb/(mu*mp)
                P_amb.convert_to_base("galactic")
                fields['pressure'] += P_amb
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
        if mode != "no_gas":
            mylog.info("Integrating gas mass profile.")
            fields["gas_mass"] = YTArray(integrate_mass(profiles["density"], rr), "Msun")

        if "stellar_density" in profiles:
            fields["stellar_density"] = YTArray(profiles["stellar_density"](rr), 
                                                "Msun/kpc**3")
            mylog.info("Integrating stellar mass profile.")
            fields["stellar_mass"] = YTArray(integrate_mass(profiles["stellar_density"], rr),
                                             "Msun")

        mdm = fields["total_mass"].copy()
        ddm = fields["total_density"].copy()
        if mode != "no_gas":
            mdm -= fields["gas_mass"]
            ddm -= fields["density"]
        if "stellar_mass" in fields:
            mdm -= fields["stellar_mass"]
            ddm -= fields["stellar_density"]
        mdm[ddm.v < 0.0][:] = mdm.max()
        ddm[ddm.v < 0.0][:] = 0.0

        if ddm.sum() > 0.0 and mdm.sum() > 0.0:
            fields["dark_matter_density"] = ddm
            fields["dark_matter_mass"] = mdm

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

    def generate_particles(self, num_particles, r_max=None, sub_sample=1):
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
            repeated to fill the required number of particles. Default: 1 
        """
        mu = self.parameters.get("mu", 1.0/muinv_default)

        mylog.info("We will be assigning %d particles." % num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(self["radius"].d,
                                                   self["gas_mass"].d,
                                                   num_particles_sub, r_max=r_max)

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(np.random.uniform(low=-1., high=1., size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields = OrderedDict()

        fields["gas", "particle_position"] = YTArray([radius*np.sin(theta)*np.cos(phi),
                                                      radius*np.sin(theta)*np.sin(phi),
                                                      radius*np.cos(theta)], "kpc").T

        mylog.info("Compute particle thermal energies, densities, and masses.")

        e_arr = 1.5*self.fields["pressure"]/self.fields["density"]
        get_energy = InterpolatedUnivariateSpline(self.fields["radius"], e_arr)

        if sub_sample > 1:
            energy = np.tile(get_energy(radius_sub), sub_sample)[:num_particles]
        else:
            energy = get_energy(radius)

        fields["gas", "thermal_energy"] = YTArray(energy, "kpc**2/Myr**2")
        fields["gas", "particle_mass"] = YTArray([mtot/num_particles]*num_particles, "Msun")

        get_density = InterpolatedUnivariateSpline(self.fields["radius"], 
                                                   self.fields["density"])

        if sub_sample > 1:
            density = np.tile(get_density(radius_sub), sub_sample)[:num_particles]
        else:
            density = get_density(radius)

        fields["gas", "density"] = YTArray(density, "Msun/kpc**3")

        mylog.info("Set particle velocities to zero.")

        fields["gas", "particle_velocity"] = YTArray(np.zeros((num_particles, 3)), "kpc/Myr")

        return ClusterParticles("gas", fields)
