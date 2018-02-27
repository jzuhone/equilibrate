import numpy as np
from yt import YTArray, mylog, get_pbar
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import \
    quad, generate_particle_radii
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cluster_particles import \
    ClusterParticles
from cluster_generator.hydrostatic import HydrostaticEquilibrium
from cluster_generator.cython_utils import generate_velocities
from collections import OrderedDict

class VirialEquilibrium(ClusterModel):

    _type_name = "virial"

    @classmethod
    def from_scratch(cls, rmin, rmax, total_profile, ptype='dark_matter',
                     num_points=1000, stellar_profile=None):

        profiles = {"total_density": total_profile}
        if stellar_profile is not None:
            profiles["stellar_density"] = stellar_profile

        hse = HydrostaticEquilibrium.from_scratch("dm_only", rmin, rmax, profiles,
                                                  num_points=num_points)
        return cls.from_hse_model(hse, ptype=ptype)

    @classmethod
    def from_hse_model(cls, hse_model, ptype='dark_matter'):
        keys = ["radius", "%s_density" % ptype, "%s_mass" % ptype,
                "gravitational_potential", "gravitational_field"]
        fields = OrderedDict([(field, hse_model[field]) for field in keys])
        parameters = {"ptype": ptype}
        return cls(hse_model.num_elements, fields, parameters=parameters)

    def __init__(self, num_elements, fields, parameters=None):
        super(VirialEquilibrium, self).__init__(num_elements, fields,
                                                parameters=parameters)
        if "distribution_function" not in self.fields:
            self._generate_df()
        else:
            f = self["distribution_function"].d[::-1]
            self.f = InterpolatedUnivariateSpline(self.ee, f)

    def _generate_df(self):
        pden = self["%s_density" % self.parameters['ptype']][::-1]
        density_spline = InterpolatedUnivariateSpline(self.ee, pden)
        g = np.zeros(self.num_elements)
        dgdp = lambda t, e: 2*density_spline(e-t*t, 1)
        pbar = get_pbar("Computing particle DF", self.num_elements)
        for i in range(self.num_elements):
            g[i] = quad(dgdp, 0., np.sqrt(self.ee[i]), epsabs=1.49e-05,
                        epsrel=1.49e-05, args=(self.ee[i]))[0]
            pbar.update(i)
        pbar.finish()
        g_spline = InterpolatedUnivariateSpline(self.ee, g)
        ff = g_spline(self.ee, 1)/(np.sqrt(8.)*np.pi**2)
        self.f = InterpolatedUnivariateSpline(self.ee, ff)
        self.fields["distribution_function"] = YTArray(ff[::-1], "Msun*Myr**3/kpc**6")

    @property
    def ee(self):
        return -self["gravitational_potential"].d[::-1]

    @property
    def ff(self):
        return self["distribution_function"].d[::-1]

    def check_model(self):
        n = self.num_elements
        rho = np.zeros(n)
        pden = self["%s_density" % self.parameters['ptype']].d
        rho_int = lambda e, psi: self.f(e)*np.sqrt(2*(psi-e))
        for i, e in enumerate(self.ee):
            rho[i] = 4.*np.pi*quad(rho_int, 0., e, args=(e,))[0]
        chk = np.abs(rho-pden)/pden
        mylog.info("The maximum relative deviation of this profile from "
                   "virial equilibrium is %g" % np.abs(chk).max())
        return rho, chk

    def generate_particles(self, num_particles, r_max=None):
        """
        Generate a set of dark matter or star particles in virial equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate 
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        """
        ptype = self.parameters["ptype"]
        key = {"dark_matter": "dm",  "stellar": "star"}[ptype]
        density = "%s_density" % ptype
        mass = "%s_mass" % ptype
        energy_spline = InterpolatedUnivariateSpline(self["radius"].d, self.ee[::-1])

        mylog.info("We will be assigning %d particles." % num_particles)
        mylog.info("Compute particle positions.")

        nonzero = self[density] > 0.0
        radius, mtot = generate_particle_radii(self["radius"].d[nonzero],
                                               self[mass].d[nonzero],
                                               num_particles, r_max=r_max)

        theta = np.arccos(np.random.uniform(low=-1., high=1., size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields = OrderedDict()

        fields[key, "particle_position"] = YTArray([radius*np.sin(theta)*np.cos(phi),
                                                    radius*np.sin(theta)*np.sin(phi),
                                                    radius*np.cos(theta)], "kpc").T

        mylog.info("Compute particle velocities.")

        psi = energy_spline(radius)
        vesc = 2.*psi
        fv2esc = vesc*self.f(psi)
        vesc = np.sqrt(vesc)

        velocity = generate_velocities(psi, vesc, fv2esc, self.f._eval_args[0],
                                       self.f._eval_args[1], self.f._eval_args[2])

        theta = np.arccos(np.random.uniform(low=-1., high=1., size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields[key, "particle_velocity"] = YTArray([velocity*np.sin(theta)*np.cos(phi),
                                                    velocity*np.sin(theta)*np.sin(phi),
                                                    velocity*np.cos(theta)], "kpc/Myr").T

        fields[key, "particle_mass"] = YTArray([mtot/num_particles]*num_particles, "Msun")
        fields[key, "particle_potential"] = -YTArray(psi, "kpc**2/Myr**2")
        fields[key, "particle_energy"] = fields[ptype, "particle_potential"]+0.5*YTArray(velocity, "kpc/Myr")**2

        return ClusterParticles(key, fields)
