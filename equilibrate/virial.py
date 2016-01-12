import numpy as np
from equilibrate.utils import \
    logspace, \
    InterpolatedUnivariateSpline, \
    unitful_zeros, \
    YTArray, pc, \
    setup_units, \
    mylog, quad, \
    integrate_toinf, \
    get_pbar, \
    integrate_mass, \
    YTQuantity
from equilibrate.equilibrium_model import EquilibriumModel
from equilibrate.cython_utils import generate_velocities
from collections import OrderedDict

class VirialEquilibrium(EquilibriumModel):

    _type_name = "virial"

    @classmethod
    def from_scratch(cls, num_particles, rmin, rmax, profile,
                     input_units=None, num_points=1000, parameters=None):

        units = setup_units(input_units)

        G = pc.G.in_units(units["G"])

        rr = logspace(rmin, rmax, num_points)

        pden = profile(rr.d)
        mylog.info("Integrating dark matter mass profile.")
        mdm = integrate_mass(profile.unitless(), rr.d)
        mylog.info("Integrating gravitational potential profile.")
        gpot_int = profile.unitless()
        gpot_profile = lambda r: gpot_int(r)*r
        gpot = G.v*(mdm/rr.d + 4.*np.pi*integrate_toinf(gpot_profile, rr.d))

        return cls(num_particles, rr, gpot, pden, mdm, units, parameters)

    @classmethod
    def from_hse_model(cls, num_particles, hse_model, parameters=None):
        if hse_model.parameters["geometry"] != "spherical":
            raise NotImplemented("VirialEquilibrium is only availabe for spherical geometries.")
        hse_model.compute_dark_matter_profiles()

        return cls(num_particles, hse_model["radius"].v,
                   -hse_model["gravitational_potential"].v,
                   hse_model["dark_matter_density"].v,
                   hse_model["dark_matter_mass"].v,
                   hse_model.units, parameters)

    def __init__(self, num_particles, rr, gpot, pden, mdm, units, parameters):

        fields = OrderedDict()

        ee = gpot[::-1]
        density_spline = InterpolatedUnivariateSpline(ee, pden[::-1])
        energy_spline = InterpolatedUnivariateSpline(rr, gpot)

        num_points = gpot.shape[0]

        g = np.zeros(num_points)
        dgdp = lambda t, e: 2*density_spline(e-t*t, 1)
        pbar = get_pbar("Computing particle DF.", num_points)
        for i in range(num_points):
            g[i] = quad(dgdp, 0., np.sqrt(ee[i]), args=(ee[i]))[0]
            pbar.update(i)
        pbar.finish()
        g_spline = InterpolatedUnivariateSpline(ee, g)
        f = lambda e: g_spline(e, 1)/(np.sqrt(8.)*np.pi**2)

        mylog.info("We will be assigning %d particles." % num_particles)

        fields["particle_position_x"] = unitful_zeros(num_particles, units["length"])
        fields["particle_position_y"] = unitful_zeros(num_particles, units["length"])
        fields["particle_position_z"] = unitful_zeros(num_particles, units["length"])
        fields["particle_velocity_x"] = unitful_zeros(num_particles, units["velocity"])
        fields["particle_velocity_y"] = unitful_zeros(num_particles, units["velocity"])
        fields["particle_velocity_z"] = unitful_zeros(num_particles, units["velocity"])

        mylog.info("Compute particle positions.")

        u = np.random.uniform(size=num_particles)
        P_r = np.insert(mdm, 0, 0.0)
        P_r /= P_r[-1]
        radius = np.interp(u, P_r, np.insert(rr, 0, 0.0), left=0.0, right=1.0)

        theta = np.arccos(np.random.uniform(low=-1.,high=1.,size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields["particle_radius"] = YTArray(radius, units["length"])
        fields["particle_position_x"][:] = radius*np.sin(theta)*np.cos(phi)
        fields["particle_position_y"][:] = radius*np.sin(theta)*np.sin(phi)
        fields["particle_position_z"][:] = radius*np.cos(theta)

        mylog.info("Compute particle velocities.")

        psi = energy_spline(radius)
        vesc = 2.*psi
        fv2esc = vesc*f(psi)
        vesc = np.sqrt(vesc)
        velocity = generate_velocities(psi, vesc, fv2esc, f)
        theta = np.arccos(np.random.uniform(low=-1.,high=1.,size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields["particle_velocity"] = YTQuantity(velocity, units["velocity"])
        fields["particle_velocity_x"][:] = velocity*np.sin(theta)*np.cos(phi)
        fields["particle_velocity_y"][:] = velocity*np.sin(theta)*np.sin(phi)
        fields["particle_velocity_z"][:] = velocity*np.cos(theta)

        fields["particle_mass"] = YTQuantity(mdm.max()/num_particles, units["mass"])
        fields["particle_potential"] = YTQuantity(psi, units["specific_energy"])
        fields["particle_energy"] = fields["particle_potential"]-0.5*fields["particle_velocity"]**2

        super(VirialEquilibrium, self).__init__(num_particles, fields, parameters, units)
