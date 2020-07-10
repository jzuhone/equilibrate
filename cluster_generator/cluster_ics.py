from cluster_generator.utils import mylog
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cluster_particles import ClusterParticles
import os
import numpy as np


def set_param(value, param_name, param):
    if param is not None:
        mylog.warning(f"'{param_name}' has already been set with value {param}. "
                      f"Overwriting with {value}!")
    return value


class ClusterICs:
    def __init__(self, num_halos, hse_files, particle_files, 
                 center, velocity, mag_file=None):
        self.num_halos = num_halos
        self.hse_files = hse_files
        self.particle_files = particle_files
        self.center = center 
        self.velocity = velocity
        self.mag_file = mag_file

    def to_file(self, filename, overwrite=False):
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")

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
                    mag_file = words[-1]
            elif len(words) == 5:
                if words[0] == "center1":
                    center[1] = np.array(words[2:]).astype("float64")

        return cls(num_halos, hse_files, particle_files, 
                   center, velocity, mag_file=mag_file)

    def setup_gamer_ics(self, input_testprob, overwrite=False):
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

    def setup_gadget_ics(self, filename, box_size, dtype='float32', overwrite=False):
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")

        hses = [ClusterModel.from_h5_file(hsef) for hsef in self.hse_files]
        parts = [ClusterParticles.from_h5_file(pf) for pf in self.particle_files]
        if self.num_halos == 1:
            parts[0].write_to_gadget_file(filename, box_size, dtype=dtype,
                                          overwrite=overwrite)
        elif self.num_halos == 2:
            pass
        else:
            pass
            