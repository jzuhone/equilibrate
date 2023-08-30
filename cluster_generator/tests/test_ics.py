from pathlib import Path
import pytest
from numpy.random import RandomState
from cluster_generator.model import ClusterModel
from cluster_generator.ics import ClusterICs, \
    compute_centers_for_binary
from cluster_generator.tests.utils import particle_answer_testing


prng = RandomState(25)
# -------------------------------------------------------------------------------------------------------------------- #
# Fixtures =========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
# Construction Testing =============================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def test_single_ics(answer_dir,gravity,answer_store):
    import os
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    ics = ClusterICs("single", 1, [f"{answer_dir}/{gravity}_model_dens_tdens.h5"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    _dir = os.getcwd()
    os.chdir(answer_dir)

    parts = ics.setup_particle_ics(prng=prng)
    os.chdir(_dir)
    particle_answer_testing(parts, f"ic_particles_{gravity}_single.h5", answer_store, answer_dir)


def test_double_ics(answer_store, answer_dir,gravity):
    import os
    num_particles = {k: 200000 for k in ["dm", "star", "gas"]}
    center1, center2 = compute_centers_for_binary([0.0, 0.0, 0.0],
                                                  3000.0, 500.0)
    velocity1 = [500.0, 0.0, 0.0]
    velocity2 = [-500.0, 0.0, 0.0]
    ics = ClusterICs("double", 2, [f"{answer_dir}/{gravity}_model_dens_tdens.h5",f"{answer_dir}/{gravity}_model_dens_tdens.h5"], [center1, center2],
                     [velocity1, velocity2], num_particles=num_particles)
    _dir = os.getcwd()
    os.chdir(answer_dir)
    parts = ics.setup_particle_ics(prng=prng)
    os.chdir(_dir)
    particle_answer_testing(parts, f"ic_particles_{gravity}_double.h5", answer_store,
                            answer_dir)
