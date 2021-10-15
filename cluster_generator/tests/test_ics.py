from cluster_generator.cluster_ics import ClusterICs
from cluster_generator.tests.utils import particle_answer_testing
from numpy.random import RandomState
from pathlib import Path


prng = RandomState(25)


def test_single_ics(answer_dir):
    p = Path(answer_dir) / "profile.h5"
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    ics = ClusterICs("single", 1, p, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    parts = ics.setup_particle_ics(prng=prng)
    particle_answer_testing(parts, "particles.h5", False, answer_dir)