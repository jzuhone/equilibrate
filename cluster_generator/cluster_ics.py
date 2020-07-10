from cluster_generator.utils import mylog
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cluster_particles import \
    ClusterParticles, \
    combine_two_clusters, \
    combine_three_clusters, \
    resample_one_cluster, \
    resample_two_clusters, \
    resample_three_clusters
import os
import numpy as np


def set_param(value, param_name, param):
    if param is not None:
        mylog.warning(f"'{param_name}' has already been set with value {param}. "
                      f"Overwriting with {value}!")
    return value


def check_for_list(x):
    if isinstance(x, list):
        return {i:v for i, v in enumerate(x)}
    else:
        return x


def format_array(x):
    return f"{x[0]} {x[1]} {x[2]}"


class ClusterICs:
    def __init__(self, num_halos, hse_files, particle_files, 
                 center, velocity, mag_file=None):
        self.num_halos = num_halos
        self.hse_files = check_for_list(hse_files)
        self.particle_files = check_for_list(particle_files)
        self.center = check_for_list(center)
        self.velocity = check_for_list(velocity)
        self.mag_file = mag_file

    def to_file(self, filename, overwrite=False):
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        outlines = [
            f"num_halos = {self.num_halos} # number of halos\n",
            f"hse_file1 = {self.hse_files[1]} # profile table of cluster 1\n",
            f"particle_file1 = {self.particle_files[1]} # particle file of cluster 1\n",
            f"center1 = {format_array(self.center[1])} # center of cluster 1 in kpc\n",
            f"velocity1 = {format_array(self.velocity[1])} # velocity of cluster 1 in km/s\n"
        ]
        if self.num_halos > 1:
            outlines += [
                f"hse_file2 = {self.hse_files[2]} # profile table of cluster 2\n",
                f"particle_file2 = {self.particle_files[2]} # particle file of cluster 2\n",
                f"center2 = {format_array(self.center[2])} # center of cluster 2 in kpc\n",
                f"velocity2 = {format_array(self.velocity[2])} # velocity of cluster 2 in km/s\n"
            ]
        if self.num_halos == 3:
            outlines += [
                f"hse_file3 = {self.hse_files[3]} # profile table of cluster 3\n",
                f"particle_file3 = {self.particle_files[3]} # particle file of cluster 3\n",
                f"center3 = {format_array(self.center[3])} # center of cluster 3 in kpc\n",
                f"velocity3 = {format_array(self.velocity[3])} # velocity of cluster 3 in km/s\n"
            ]
        with open(filename, "w") as f:
            f.writelines(outlines)

    @classmethod
    def from_file(cls, filename):
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        num_halos = None
        hse_files = {1: None, 2: None, 3: None}
        particle_files = {1: None, 2: None, 3: None}
        center = {1: None, 2: None, 3: None}
        velocity = {1: None, 2: None, 3: None}
        mag_file = None
        for line in lines:
            words = line.strip().split()
            if words[0].startswith("#"):
                continue
            elif len(words) == 3:
                if words[0] == "num_halos":
                    num_halos = set_param("num_halos", int(words[2]), num_halos)
                elif words[0] == "hse_file1":
                    hse_files[1] = set_param("hse_file1", words[2], hse_files[1])
                elif words[0] == "hse_file2" and num_halos > 1:
                    hse_files[2] = set_param("hse_file2", words[2], hse_files[2])
                elif words[0] == "hse_file3" and num_halos > 2:
                    hse_files[3] = set_param("hse_file3", words[2], hse_files[3])
                elif words[0] == "hse_file1":
                    hse_files[1] = set_param("hse_file1", words[2], hse_files[1])
                elif words[0] == "hse_file2" and num_halos > 1:
                    hse_files[2] = set_param("hse_file2", words[2], hse_files[2])
                elif words[0] == "hse_file3" and num_halos > 2:
                    hse_files[3] = set_param("hse_file3", words[2], hse_files[3])
                elif words[0] == "mag_file":
                    mag_file = set_param("mag_file", words[2], mag_file)
            elif len(words) == 5:
                if words[0] == "center1":
                    center[1] = set_param(
                        "center1", np.array(words[2:]).astype("float64"),
                        center[1])
                if words[0] == "center2":
                    center[2] = set_param(
                        "center2", np.array(words[2:]).astype("float64"),
                        center[2])
                if words[0] == "center3":
                    center[3] = set_param(
                        "center3", np.array(words[2:]).astype("float64"),
                        center[3])
                if words[0] == "velocity1":
                    velocity[1] = set_param(
                        "velocity1", np.array(words[2:]).astype("float64"),
                        velocity[1])
                if words[0] == "velocity2":
                    velocity[2] = set_param(
                        "velocity2", np.array(words[2:]).astype("float64"),
                        velocity[2])
                if words[0] == "velocity3":
                    velocity[3] = set_param(
                        "velocity3", np.array(words[2:]).astype("float64"),
                        velocity[3])

        return cls(num_halos, hse_files, particle_files,
                   center, velocity, mag_file=mag_file)

    def setup_gamer_ics(self, input_testprob, overwrite=False):
        if os.path.exists(input_testprob) and not overwrite:
            raise RuntimeError(f"{input_testprob} exists and overwrite=False!")
        outlines = [
            "# problem-specific runtime parameters\n",
            f"Merger_Coll_NumHalos    {self.num_halos} # number of halos\n",
            f"Merger_File_Prof1       {self.hse_files[1]} # profile table of cluster 1\n",
            f"Merger_File_Par1        {self.particle_files[1]} # particle file of cluster 1\n",
            f"Merger_Coll_PosX1       {self.center[1][0]} # X-center of cluster 1 in kpc\n",
            f"Merger_Coll_PosY1       {self.center[1][1]} # Y-center of cluster 1 in kpc\n",
            f"Merger_Coll_VelX1       {self.velocity[1][0]} # X-velocity of cluster 1 in km/s\n",
            f"Merger_Coll_VelY1       {self.velocity[1][1]} # Y-velocity of cluster 1 in km/s\n",
        ]
        if self.num_halos > 1:
            outlines += [
                f"Merger_File_Prof2       {self.hse_files[2]} # profile table of cluster 2\n",
                f"Merger_File_Par2        {self.particle_files[2]} # particle file of cluster 2\n",
                f"Merger_Coll_PosX2       {self.center[2][0]} # X-center of cluster 2 in kpc\n",
                f"Merger_Coll_PosY2       {self.center[2][1]} # Y-center of cluster 2 in kpc\n",
                f"Merger_Coll_VelX2       {self.velocity[2][0]} # X-velocity of cluster 2 in km/s\n",
                f"Merger_Coll_VelY2       {self.velocity[2][1]} # Y-velocity of cluster 2 in km/s\n",
            ]
        if self.num_halos == 3:
            outlines += [
                f"Merger_File_Prof3       {self.hse_files[3]} # profile table of cluster 3\n",
                f"Merger_File_Par3        {self.particle_files[3]} # particle file of cluster 3\n",
                f"Merger_Coll_PosX3       {self.center[3][0]} # X-center of cluster 3 in kpc\n",
                f"Merger_Coll_PosY3       {self.center[3][1]} # Y-center of cluster 3 in kpc\n",
                f"Merger_Coll_VelX3       {self.velocity[3][0]} # X-velocity of cluster 3 in km/s\n",
                f"Merger_Coll_VelY3       {self.velocity[3][1]} # Y-velocity of cluster 3 in km/s\n",
            ]
        with open(input_testprob, "w") as f:
            f.writelines(outlines)
        if self.mag_file is not None:
            mylog.info(f"Rename the file '{self.mag_file}' to 'B_IC' "
                       f"and place it in the same directory as the "
                       f"Input__* files.")

    def setup_gadget_ics(self, filename, box_size, dtype='float32', 
                         overwrite=False):
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        parts = [ClusterParticles.from_h5_file(pf) for pf in self.particle_files]
        if self.num_halos == 1:
            all_parts = parts[0]
        elif self.num_halos == 2:
            all_parts = combine_two_clusters(parts[0], parts[1], hses[0],
                                             hses[1], self.center[1], 
                                             self.center[2], self.velocity[1],
                                             self.velocity[2])
        else:
            all_parts = combine_three_clusters(parts[0], parts[1], parts[2],
                                               hses[0], hses[1], hses[2], 
                                               self.center[1], self.center[2], 
                                               self.center[3], self.velocity[1],
                                               self.velocity[2], self.velocity[3])
        all_parts.write_to_gadget_file(filename, box_size, dtype=dtype,
                                       overwrite=overwrite)

    def resample_gadget_ics(self, infile, outfile, box_size,
                            dtype='float32', r_max=5000.0,
                            overwrite=False):
        if os.path.exists(outfile) and not overwrite:
            raise RuntimeError(f"{outfile} exists and overwrite=False!")
        hses = [ClusterModel.from_h5_file(hf) for hf in self.hse_files]
        parts = ClusterParticles.from_gadget_file(infile)
        if self.num_halos == 1:
            new_parts = resample_one_cluster(parts, hses[0], self.center[1],
                                             self.velocity[1])
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(parts, hses[0], hses[1],
                                              self.center[1], self.center[2],
                                              self.velocity[1], self.velocity[2],
                                              [r_max]*2)
        else:
            new_parts = resample_three_clusters(parts, hses[0], hses[1], hses[2],
                                                self.center[1], self.center[2],
                                                self.center[3], self.velocity[1], 
                                                self.velocity[2], self.velocity[3], 
                                                [r_max]*3)
        new_parts.write_to_gadget_file(outfile, box_size, dtype=dtype,
                                       overwrite=overwrite)
