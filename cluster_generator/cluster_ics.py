from cluster_generator.utils import mylog, ensure_ytarray
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


class ClusterICs:
    def __init__(self, num_halos, hse_files, center,
                 velocity, num_particles=None, mag_file=None, 
                 resample_file=None):
        self.num_halos = num_halos
        self.hse_files = hse_files
        self.center = ensure_ytarray(center, "kpc")
        self.velocity = ensure_ytarray(velocity, "kpc/Myr")
        self.mag_file = mag_file
        self.resample_file = resample_file
        if num_particles is None:
            self.tot_np = {"dm": 0, "gas": 0, "star": 0}
        else:
            self.tot_np = num_particles
        self._determine_num_particles()

    def _determine_num_particles(self):
        from collections import defaultdict
        dm_masses = []
        gas_masses = []
        star_masses = []
        for hsef in self.hse_files:
            hse = ClusterModel.from_h5_file(hsef)
            dm_masses.append(hse["dark_matter_mass"][-1].value)
            if "gas_mass" in hse:
                gmass = hse["gas_mass"][-1].value
            else:
                gmass = 0.0
            gas_masses.append(gmass)
            if "stellar_mass" in hse:
                smass = hse["stellar_mass"][-1].value
            else:
                smass = 0.0
            star_masses.append(smass)
        tot_dm_mass = np.sum(dm_masses)
        tot_gas_mass = np.sum(gas_masses)
        tot_star_mass = np.sum(star_masses)
        self.num_particles = defaultdict(list)
        for i in range(self.num_halos):
            if self.tot_np["dm"] > 0:
                ndp = int(self.tot_np["dm"]*dm_masses[i]/tot_dm_mass)
            else:
                ndp = 0
            self.num_particles["dm"].append(ndp)
            if self.tot_np["gas"] > 0:
                ngp = int(self.tot_np["gas"]*gas_masses[i]/tot_gas_mass)
            else:
                ngp = 0
            self.num_particles["gas"].append(ngp)
            if self.tot_np["star"] > 0:
                nsp = int(self.tot_np["star"]*star_masses[i]/tot_star_mass)
            else:
                nsp = 0
            self.num_particles["star"].append(nsp)

    def _generate_particles(self, r_max=None):
        parts = []
        if self.num_halos == 1:
            r_max = None
        for i, hf in enumerate(self.hse_files):
            hse = ClusterModel.from_h5_file(hf)
            vird = VirialEquilibrium.from_hse_model(hse, ptype="dark_matter")
            p = vird.generate_particles(
                self.num_particles["dm"][i], r_max=r_max)
            if self.num_particles["star"][i] > 0:
                virs = VirialEquilibrium.from_hse_model(hse, ptype="stellar")
                sp = virs.generate_particles(
                    self.num_particles["star"][i], r_max=r_max)
                p = p + sp
            if self.num_particles["gas"][i] > 0:
                gp = hse.generate_particles(
                    self.num_particles["gas"][i], r_max=r_max)
                p = p + gp
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
        out["num_halos"] = self.num_halos
        out.yaml_add_eol_comment("number of halos", key='num_halos')
        out["profile1"] = self.hse_files[0]
        out.yaml_add_eol_comment("profile for cluster 1", key='profile1')
        out["center1"] = self.center[0].tolist()
        out.yaml_add_eol_comment("center for cluster 1", key='center1')
        out["velocity1"] = self.velocity[0].tolist()
        out.yaml_add_eol_comment("velocity for cluster 1", key='velocity1')
        if self.num_halos > 1:
            out["profile2"] = self.hse_files[1]
            out.yaml_add_eol_comment("profile for cluster 2", key='profile2')
            out["center2"] = self.center[1].tolist()
            out.yaml_add_eol_comment("center for cluster 2", key='center2')
            out["velocity2"] = self.velocity[1].tolist()
            out.yaml_add_eol_comment("velocity for cluster 2", key='velocity2')
        if self.num_halos == 3:
            out["profile3"] = self.hse_files[2]
            out.yaml_add_eol_comment("profile for cluster 3", key='profile3')
            out["center3"] = self.center[2].tolist()
            out.yaml_add_eol_comment("center for cluster 3", key='center3')
            out["velocity3"] = self.velocity[2].tolist()
            out.yaml_add_eol_comment("velocity for cluster 3", key='velocity3')
        if self.tot_np["dm"] > 0:
            out["num_dm_particles"] = self.tot_np["dm"]
            out.yaml_add_eol_comment("number of DM particles", key='num_dm_particles')
        if self.tot_np["gas"] > 0:
            out["num_gas_particles"] = self.tot_np["gas"]
            out.yaml_add_eol_comment("number of gas particles", key='num_gas_particles')
        if self.tot_np["star"] > 0:
            out["num_star_particles"] = self.tot_np["star"]
            out.yaml_add_eol_comment("number of star particles", key='num_star_particles')
        if self.mag_file is not None:
            out["mag_file"] = self.mag_file
            out.yaml_add_eol_comment("3D magnetic field file", key='mag_file')
        if self.resample_file is not None:
            out["resample_file"] = self.resample_file
            out.yaml_add_eol_comment("Gadget resampling file", key='resample_file')
        yaml = YAML()
        with open(filename, "w") as f:
            yaml.dump(out, f)

    @classmethod
    def from_file(cls, filename):
        r"""
        Read the initial conditions information
        from a YAML-formatted *filename*.
        """
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(filename, "r") as f:
            params = yaml.load(f)
        num_halos = params["num_halos"]
        hse_files = [params[f"profile{i}"] for i in range(1, 4)]
        center = [np.array(params[f"center{i}"]) for i in range(1, 4)]
        velocity = [np.array(params[f"velocity{i}"]) for i in range(1, 4)]
        num_particles = {k: params.get(f"num_{k}_particles", 0)
                         for k in ["gas", "dm", "star"]}
        mag_file = params.get("mag_file", None)
        resample_file = params.get("resample_file", None)
        return cls(num_halos, hse_files, center, velocity, 
                   num_particles=num_particles, mag_file=mag_file, 
                   resample_file=resample_file)

    def setup_gamer_ics(self, input_testprob, particle_file_prefix,
                        overwrite=False):
        r"""

        Write the "Input_TestProb" file for use with the 
        ClusterMerger setup in GAMER.

        Parameters
        ----------
        input_testprob : string 
            The path to the Input__TestProb file which will
            include the parameters. 
        overwrite : boolean, optional
            If True, a file of the same name will be overwritten.
            Default: False
        """
        if os.path.exists(input_testprob) and not overwrite:
            raise RuntimeError(f"{input_testprob} exists and overwrite=False!")
        parts = self._generate_particles()
        outlines = [
            "# problem-specific runtime parameters\n",
            f"Merger_Coll_NumHalos    {self.num_halos} # number of halos\n"
        ]
        for i in range(self.num_halos):
            particle_file = f"{particle_file_prefix}_parts{i}.h5"
            parts[i].write_gamer_input(particle_file)
            outlines += [
                f"Merger_File_Prof{i+1}       {self.hse_files[i]} # profile table of cluster {i}\n",
                f"Merger_File_Par{i+1}        {particle_file} # particle file of cluster {i}\n",
                f"Merger_Coll_PosX{i+1}       {self.center[i][0]} # X-center of cluster {i} in kpc\n",
                f"Merger_Coll_PosY{i+1}       {self.center[i][1]} # Y-center of cluster {i} in kpc\n",
                f"Merger_Coll_VelX{i+1}       {self.velocity[i][0]} # X-velocity of cluster {i} in km/s\n",
                f"Merger_Coll_VelY{i+1}       {self.velocity[i][1]} # Y-velocity of cluster {i} in km/s\n",
            ]
        with open(input_testprob, "w") as f:
            f.writelines(outlines)
        num_particles = sum([self.tot_np[key] for key in ["dm", "star"]])
        mylog.info(f"In the Input__Parameter file, set PAR__NPAR = {num_particles}.")
        if self.mag_file is not None:
            mylog.info(f"Rename the file '{self.mag_file}' to 'B_IC' "
                       f"and place it in the same directory as the "
                       f"Input__* files, and set OPT__INIT_BFIELD_BYFILE "
                       f"to 1 in Input__Parameter")

    def setup_particle_ics(self, r_max=5000.0):
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
        r_max : float, optional
            The maximum radius in kpc for each cluster to which the 
            resampling from the profiles occurs. Ignored in the case
            of a single cluster. Default: 5000.0
        """
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        parts = self._generate_particles(r_max=r_max)
        if self.num_halos == 1:
            all_parts = parts[0]
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

    def resample_gadget_ics(self, filename, dtype='float32', r_max=5000.0,
                            overwrite=False):
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
        dtype : string, optional
            The datatype of the fields to be written. Default: float32
        r_max : float, optional
            The maximum radius in kpc for each cluster to which the 
            resampling from the profiles occurs. Ignored in the case
            of a single cluster. Default: 5000.0
        overwrite : boolean, optional
            If True, a file of the same name will be overwritten.
            Default: False
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        if self.resample_file is None or not os.path.exists(self.resample_file):
            raise IOError("A 'resample_file' must be specified in the "
                          "parameter file for this operation!")
        parts = ClusterParticles.from_gadget_file(self.resample_file)
        if self.num_halos == 1:
            new_parts = resample_one_cluster(parts, hses[0], self.center[0],
                                             self.velocity[0])
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(parts, hses[0], hses[1],
                                              self.center[0], self.center[1],
                                              self.velocity[0], self.velocity[1],
                                              [r_max]*2)
        else:
            new_parts = resample_three_clusters(parts, hses[0], hses[1], hses[2],
                                                self.center[0], self.center[1],
                                                self.center[2], self.velocity[0], 
                                                self.velocity[1], self.velocity[2], 
                                                [r_max]*3)
        new_parts.write_to_gadget_file(filename, parts.box_size, dtype=dtype,
                                       overwrite=overwrite)

    def setup_flash_ics(self):
        pass

    def setup_athena_ics(self):
        pass
