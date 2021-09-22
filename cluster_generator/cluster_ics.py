from cluster_generator.utils import ensure_ytarray, ensure_list
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.virial import VirialEquilibrium
from cluster_generator.cluster_particles import \
    ClusterParticles, \
    combine_two_clusters, \
    combine_three_clusters, \
    resample_one_cluster, \
    resample_two_clusters, \
    resample_three_clusters
import os
import numpy as np
from ruamel.yaml import YAML


def compute_centers_for_binary(center, d, b, a=0.0):
    """
    Given a common center and distance parameters, calculate the
    central positions of two clusters.

    First, the separation along the x-direction is determined
    by:

    sep_x = sqrt(d**2-b**2-a**2)

    where d is the distance between the two clusters, b is the
    impact parameter in the y-direction, and a is the impact 
    parameter in the z-direction. So the resulting centers are
    calculated as:

    center1 = [center-0.5*sep_x, center-0.5*b, center-0.5*a]
    center2 = [center+0.5*sep_x, center+0.5*b, center+0.5*a]

    Parameters
    ----------
    center : array-like
        The center from which the distance parameters for
        the two clusters will be calculated.
    d : float
        The distance between the two clusters.
    b : float
        The impact parameter in the y-direction, in kpc.
    a : float, optional
        The impact parameter in the z-direction, in kpc.
        Default: 0.0
    """
    d = np.sqrt(d*d-b*b-a*a)
    diff = np.array([d, b, a])
    center1 = center - 0.5*diff
    center2 = center + 0.5*diff
    return center1, center2


class ClusterICs:
    def __init__(self, basename, num_halos, hse_files, center,
                 velocity, num_particles=None, mag_file=None, 
                 particle_files=None, r_max=10000.0):
        self.basename = basename
        self.num_halos = num_halos
        self.hse_files = ensure_list(hse_files)
        self.center = ensure_ytarray(center, "kpc")
        self.velocity = ensure_ytarray(velocity, "kpc/Myr")
        if self.num_halos == 1:
            self.center = self.center.reshape(1, 3)
            self.velocity = self.velocity.reshape(1, 3)
        self.mag_file = mag_file
        self.r_max = r_max
        if num_particles is None:
            self.tot_np = {"dm": 0, "gas": 0, "star": 0}
        else:
            self.tot_np = num_particles
        self._determine_num_particles()
        self.particle_files = [None]*3
        if particle_files is not None:
            self.particle_files[:num_halos] = particle_files[:]

    _profiles = None

    @property
    def profiles(self):
        if self._profiles is None:
            self._profiles = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        return self._profiles

    def _determine_num_particles(self):
        from collections import defaultdict
        dm_masses = []
        gas_masses = []
        star_masses = []
        for hsef in self.hse_files:
            hse = ClusterModel.from_h5_file(hsef)
            idxs = hse["radius"] < self.r_max
            dm_masses.append(hse["dark_matter_mass"][idxs][-1].value)
            if "gas_mass" in hse:
                gmass = hse["gas_mass"][idxs][-1].value
            else:
                gmass = 0.0
            gas_masses.append(gmass)
            if "stellar_mass" in hse:
                smass = hse["stellar_mass"][idxs][-1].value
            else:
                smass = 0.0
            star_masses.append(smass)
        tot_dm_mass = np.sum(dm_masses)
        tot_gas_mass = np.sum(gas_masses)
        tot_star_mass = np.sum(star_masses)
        self.num_particles = defaultdict(list)
        for i in range(self.num_halos):
            if self.tot_np.get("dm", 0) > 0:
                ndp = np.rint(
                    self.tot_np["dm"]*dm_masses[i]/tot_dm_mass).astype("int")
            else:
                ndp = 0
            self.num_particles["dm"].append(ndp)
            if self.tot_np.get("gas", 0) > 0:
                ngp = np.rint(
                    self.tot_np["gas"]*gas_masses[i]/tot_gas_mass).astype("int")
            else:
                ngp = 0
            self.num_particles["gas"].append(ngp)
            if self.tot_np.get("star", 0) > 0:
                nsp = np.rint(
                    self.tot_np["star"]*star_masses[i]/tot_star_mass).astype("int")
            else:
                nsp = 0
            self.num_particles["star"].append(nsp)

    def _generate_particles(self, regenerate_particles=False):
        parts = []
        for i, hf in enumerate(self.hse_files):
            if regenerate_particles or self.particle_files[i] is None:
                hse = ClusterModel.from_h5_file(hf)
                vird = VirialEquilibrium.from_hse_model(hse, ptype="dark_matter")
                p = vird.generate_particles(
                    self.num_particles["dm"][i], r_max=self.r_max)
                if self.num_particles["star"][i] > 0:
                    virs = VirialEquilibrium.from_hse_model(hse, ptype="stellar")
                    sp = virs.generate_particles(
                        self.num_particles["star"][i], r_max=self.r_max)
                    p = p + sp
                if self.num_particles["gas"][i] > 0:
                    gp = hse.generate_particles(
                        self.num_particles["gas"][i], r_max=self.r_max)
                    p = p + gp
                parts.append(p)
                outfile = f"{self.basename}_{i}_particles.h5"
                p.write_particles_to_h5(outfile, overwrite=True)
                self.particle_files[i] = outfile
            else:
                p = ClusterParticles.from_h5_file(self.particle_files[i])
                parts.append(p)
        return parts

    def to_file(self, filename, overwrite=False):
        r"""
        Write the initial conditions information to a file.

        Parameters
        ----------
        filename : string
            The file to write the initial conditions information to.
        overwrite : boolean, optional
            If True, overwrite a file with the same name. Default: False
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        from ruamel.yaml.comments import CommentedMap
        out = CommentedMap()
        out["basename"] = self.basename
        out.yaml_add_eol_comment("base name for ICs", key="basename")
        out["num_halos"] = self.num_halos
        out.yaml_add_eol_comment("number of halos", key='num_halos')
        out["profile1"] = self.hse_files[0]
        out.yaml_add_eol_comment("profile for cluster 1", key='profile1')
        out["center1"] = self.center[0].tolist()
        out.yaml_add_eol_comment("center for cluster 1", key='center1')
        out["velocity1"] = self.velocity[0].tolist()
        out.yaml_add_eol_comment("velocity for cluster 1", key='velocity1')
        if self.particle_files[0] is not None:
            out["particle_file1"] = self.particle_files[0]
            out.yaml_add_eol_comment("particle file for cluster 1",
                                     key='particle_file1')
        if self.num_halos > 1:
            out["profile2"] = self.hse_files[1]
            out.yaml_add_eol_comment("profile for cluster 2", key='profile2')
            out["center2"] = self.center[1].tolist()
            out.yaml_add_eol_comment("center for cluster 2", key='center2')
            out["velocity2"] = self.velocity[1].tolist()
            out.yaml_add_eol_comment("velocity for cluster 2", key='velocity2')
            if self.particle_files[1] is not None:
                out["particle_file2"] = self.particle_files[1]
                out.yaml_add_eol_comment("particle file for cluster 2", 
                                         key='particle_file2')
        if self.num_halos == 3:
            out["profile3"] = self.hse_files[2]
            out.yaml_add_eol_comment("profile for cluster 3", key='profile3')
            out["center3"] = self.center[2].tolist()
            out.yaml_add_eol_comment("center for cluster 3", key='center3')
            out["velocity3"] = self.velocity[2].tolist()
            out.yaml_add_eol_comment("velocity for cluster 3", key='velocity3')
            if self.particle_files[2] is not None:
                out["particle_file3"] = self.particle_files[2]
                out.yaml_add_eol_comment("particle file for cluster 3",
                                         key='particle_file3')
        if self.tot_np.get("dm", 0) > 0:
            out["num_dm_particles"] = self.tot_np["dm"]
            out.yaml_add_eol_comment("number of DM particles", 
                                     key='num_dm_particles')
        if self.tot_np.get("gas", 0) > 0:
            out["num_gas_particles"] = self.tot_np["gas"]
            out.yaml_add_eol_comment("number of gas particles", 
                                     key='num_gas_particles')
        if self.tot_np.get("star", 0) > 0:
            out["num_star_particles"] = self.tot_np["star"]
            out.yaml_add_eol_comment("number of star particles", 
                                     key='num_star_particles')
        if self.mag_file is not None:
            out["mag_file"] = self.mag_file
            out.yaml_add_eol_comment("3D magnetic field file", key='mag_file')
        out["r_max"] = self.r_max
        out.yaml_add_eol_comment("Maximum radius of particles", key='r_max')
        yaml = YAML()
        with open(filename, "w") as f:
            yaml.dump(out, f)

    @classmethod
    def from_file(cls, filename):
        r"""
        Read the initial conditions information
        from a YAML-formatted `filename`.
        """
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(filename, "r") as f:
            params = yaml.load(f)
        basename = params["basename"]
        num_halos = params["num_halos"]
        hse_files = [params[f"profile{i}"] for i in range(1, num_halos+1)]
        center = [np.array(params[f"center{i}"]) for i in range(1, num_halos+1)]
        velocity = [np.array(params[f"velocity{i}"]) 
                    for i in range(1, num_halos+1)]
        num_particles = {k: params.get(f"num_{k}_particles", 0)
                         for k in ["gas", "dm", "star"]}
        mag_file = params.get("mag_file", None)
        particle_files = [params.get(f"particle_file{i}", None)
                          for i in range(1, num_halos+1)]
        r_max = params.get("r_max", 10000.0)
        return cls(basename, num_halos, hse_files, center, velocity,
                   num_particles=num_particles, mag_file=mag_file,
                   particle_files=particle_files, r_max=r_max)

    def setup_particle_ics(self, regenerate_particles=False):
        r"""
        From a set of cluster models and their relative positions and
        velocities, set up initial conditions for use with SPH codes.

        This routine will either generate a single cluster or will combine
        two or three clusters together. If more than one cluster is 
        generated, the gas particles will have their densities set by 
        adding the densities from the overlap of the two particles 
        together, and will have their thermal energies and velocities 
        set by mass-weighting them from the two profiles.

        Parameters
        ----------
        """
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        parts = self._generate_particles(
            regenerate_particles=regenerate_particles)
        if self.num_halos == 1:
            all_parts = parts[0]
            all_parts.add_offsets(self.center[0], self.velocity[0])
        elif self.num_halos == 2:
            all_parts = combine_two_clusters(parts[0], parts[1], hses[0],
                                             hses[1], self.center[0],
                                             self.center[1], self.velocity[0],
                                             self.velocity[1])
        else:
            all_parts = combine_three_clusters(parts[0], parts[1], parts[2],
                                               hses[0], hses[1], hses[2],
                                               self.center[0], self.center[1],
                                               self.center[2], self.velocity[0],
                                               self.velocity[1], self.velocity[2])
        return all_parts

    def resample_particle_ics(self, parts, passive_scalars=None):
        r"""
        Given a Gadget-HDF5-like initial conditions file which has been
        output from some type of relaxation process (such as making a 
        glass or using MESHRELAX in the case of Arepo), resample the density,
        thermal energy, and velocity fields onto the gas particles/cells from
        the initial hydrostatic profiles.

        Parameters
        ----------
        filename : string
            The name of file to output the resampled ICs to.
        """
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        if self.num_halos == 1:
            new_parts = resample_one_cluster(parts, hses[0], self.center[0],
                                             self.velocity[0])
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(parts, hses[0], hses[1],
                                              self.center[0], self.center[1],
                                              self.velocity[0], self.velocity[1],
                                              [self.r_max]*2,
                                              passive_scalars=passive_scalars)
        else:
            new_parts = resample_three_clusters(parts, hses[0], hses[1], hses[2],
                                                self.center[0], self.center[1],
                                                self.center[2], self.velocity[0], 
                                                self.velocity[1], self.velocity[2], 
                                                [self.r_max]*3, 
                                                passive_scalars=passive_scalars)
        return new_parts
