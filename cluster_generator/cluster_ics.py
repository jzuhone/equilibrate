from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cluster_particles import ClusterParticles

class ClusterICs:
    def __init__(self, num_halos, hse_files, particle_files, 
                 center, velocity):
        self.num_halos = num_halos
        self.hse_files = hse_files
        self.particle_files = particle_files
        self.center = center 
        self.velocity = velocity

    @classmethod
    def from_file(cls, filename):
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        num_halos = None
        hse_files = []
        particle_files = []
        center = []
        velocity = []
        for line in lines:
            words = line.strip().split()
            if words[0] == "num_halos":
                num_halos = int(words[-1])
            if words[0] == "hse1":
                hse_files.append(words[-1])
            if words[0] == "hse2" and num_halos > 1:
                hse_files.append(words[-1])
            if words[0] == "hse3" and num_halos > 2:
                hse_files.append(words[-1])
            if words[0] == "particles1":
                particle_files.append(words[-1])
            if words[0] == "particles2" and num_halos > 1:
                particle_files.append(words[-1])
            if words[0] == "particles3" and num_halos > 2:
                particle_files.append(words[-1])

        return cls(num_halos, hse_files, particle_files, 
                   center, velocity)
        
    def setup_gadget_ics(self, outfile, box_size, dtype='float32', overwrite=False):
        hses = [ClusterModel.from_h5_file(hsef) for hsef in self.hse_files]
        parts = [ClusterParticles.from_h5_file(pf) for pf in self.particle_files]
        if self.num_halos == 1:
            parts[0].write_to_gadget_file(outfile, box_size, dtype=dtype, 
                                          overwrite=overwrite)
        elif self.num_halos == 2:
            pass
        else:
            pass
            