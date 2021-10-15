from cluster_generator.cluster_ics import ClusterICs, \
    compute_centers_for_binary
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


def test_double_ics(answer_store, answer_dir):
    p = Path(answer_dir) / "profile.h5"
    num_particles = {k: 200000 for k in ["dm", "star", "gas"]}
    center1, center2 = compute_centers_for_binary([0.0, 0.0, 0.0], 
                                                  3000.0, 500.0)
    velocity1 = [500.0, 0.0, 0.0]
    velocity2 = [-500.0, 0.0, 0.0]
    ics = ClusterICs("double", 2, [p, p], [center1, center2],
                     [velocity1, velocity2], num_particles=num_particles)
    parts = ics.setup_particle_ics(prng=prng)
    particle_answer_testing(parts, "double_particles.h5", answer_store, 
                            answer_dir)
