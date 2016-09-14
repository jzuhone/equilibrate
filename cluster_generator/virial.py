import numpy as np
from yt import YTArray, YTQuantity, mylog, get_pbar
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import \
    G, quad, \
    integrate_toinf, \
    integrate_mass
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cython_utils import generate_velocities
from collections import OrderedDict

class VirialEquilibrium(ClusterModel):

    _type_name = "virial"

    @classmethod
    def from_scratch(cls, num_particles, rmin, rmax, profile,
                     num_points=1000):

        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)

        pden = profile(rr)
        mylog.info("Integrating dark matter mass profile.")
        mdm = integrate_mass(profile, rr)
        mylog.info("Integrating gravitational potential profile.")
        gpot_profile = lambda r: profile(r)*r
        gpot = G.v*(mdm/rr + 4.*np.pi*integrate_toinf(gpot_profile, rr))

        return cls(num_particles, rr, gpot, pden, mdm)

    @classmethod
    def from_hse_model(cls, num_particles, hse_model):
        if hse_model.geometry != "spherical":
            raise NotImplemented("VirialEquilibrium is only available for spherical geometries.")
        gpot = -hse_model["gravitational_potential"].in_units("kpc**2/Myr**2")
        return cls(num_particles, hse_model["radius"].v,
                   gpot.v, hse_model["dark_matter_density"].v,
                   hse_model["dark_matter_mass"].v)

    def __init__(self, num_particles, rr, gpot, pden, mdm):

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

        self.ee = ee
        self.f = f
        self.rr = rr
        self.pden = pden

        mylog.info("We will be assigning %d particles." % num_particles)
        mylog.info("Compute particle positions.")

        u = np.random.uniform(size=num_particles)
        P_r = np.insert(mdm, 0, 0.0)
        P_r /= P_r[-1]
        radius = np.interp(u, P_r, np.insert(rr, 0, 0.0), left=0.0, right=1.0)

        theta = np.arccos(np.random.uniform(low=-1.,high=1.,size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields["particle_radius"] = YTArray(radius, "kpc")
        fields["particle_position_x"] = YTArray(radius*np.sin(theta)*np.cos(phi), "kpc")
        fields["particle_position_y"] = YTArray(radius*np.sin(theta)*np.sin(phi), "kpc")
        fields["particle_position_z"] = YTArray(radius*np.cos(theta), "kpc")

        mylog.info("Compute particle velocities.")

        psi = energy_spline(radius)
        vesc = 2.*psi
        fv2esc = vesc*f(psi)
        vesc = np.sqrt(vesc)
        velocity = generate_velocities(psi, vesc, fv2esc, f)
        theta = np.arccos(np.random.uniform(low=-1.,high=1.,size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields["particle_velocity"] = YTArray(velocity, "kpc/Myr")
        fields["particle_velocity_x"] = YTArray(velocity*np.sin(theta)*np.cos(phi), "kpc/Myr")
        fields["particle_velocity_y"] = YTArray(velocity*np.sin(theta)*np.sin(phi), "kpc/Myr")
        fields["particle_velocity_z"] = YTArray(velocity*np.cos(theta), "kpc/Myr")

        fields["particle_mass"] = YTArray([mdm.max()/num_particles], "Msun")
        fields["particle_potential"] = YTArray(psi, "kpc**2/Myr**2")
        fields["particle_energy"] = fields["particle_potential"]-0.5*fields["particle_velocity"]**2

        super(VirialEquilibrium, self).__init__(num_particles, fields, "spherical")

    def check_model(self):
        n = len(self.ee)
        rho = np.zeros(n)
        rho_int = lambda e, psi: self.f(e)*np.sqrt(2*(psi-e))
        for i, e in enumerate(self.ee[::-1]):
            rho[i] = 4.*np.pi*quad(rho_int, 0., e, args=(e))[0]
        chk = np.abs(rho-self.pden)/self.pden
        mylog.info("The maximum relative deviation of this profile from "
                   "virial equilibrium is %g" % np.abs(chk).max())
        return rho, chk