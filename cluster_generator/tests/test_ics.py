from pathlib import Path
import pytest
from numpy.random import RandomState
from cluster_generator.model import ClusterModel
from cluster_generator.ics import ClusterICs, \
    compute_centers_for_binary
import cluster_generator.codes as cds
from cluster_generator.tests.utils import particle_answer_testing, generate_model_dens_tdens


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

# -------------------------------------------------------------------------------------------------------------------- #
# Code Specific Tests ================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
def test_amr_ics(answer_dir,gravity,answer_store):
    import os
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    ics = ClusterICs("single", 1, [f"{answer_dir}/{gravity}_model_dens_tdens.h5"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    _dir = os.getcwd()
    os.chdir(answer_dir)
    ps = ics.setup_particle_ics()
    parts = cds.write_amr_particles(ps,f"{answer_dir}/amr_parts.h5",["dm","star"],{"dm":1,"star":2})

def test_gamer_ics(answer_dir,gravity,answer_store):
    import os
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    loc = os.getcwd()
    os.chdir(answer_dir)
    if not os.path.exists(f"{gravity}_model_dens_tdens.h5"):
        m = generate_model_dens_tdens(gravity=gravity)
        m.write_model_to_h5(f"{gravity}_model_dens_tdens.h5")

    ics = ClusterICs("single", 1, [f"{gravity}_model_dens_tdens.h5"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    cds.setup_gamer_ics(ics)
    os.chdir(loc)

def test_flash_ics(answer_dir,gravity,answer_store):
    import os
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    loc = os.getcwd()
    os.chdir(answer_dir)
    if not os.path.exists(f"{gravity}_model_dens_tdens.h5"):
        m = generate_model_dens_tdens(gravity=gravity)
        m.write_model_to_h5(f"{gravity}_model_dens_tdens.h5")

    ics = ClusterICs("single", 1, [f"{gravity}_model_dens_tdens.h5"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    cds.setup_flash_ics(ics)
    os.chdir(loc)
def test_ramses_ics(answer_dir,gravity,answer_store):
    import os
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    loc = os.getcwd()
    os.chdir(answer_dir)
    if not os.path.exists(f"{gravity}_model_dens_tdens.h5"):
        m = generate_model_dens_tdens(gravity=gravity)
        m.write_model_to_h5(f"{gravity}_model_dens_tdens.h5")

    ics = ClusterICs("single", 1, [f"{gravity}_model_dens_tdens.h5"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    cds.setup_ramses_ics(ics)
    os.chdir(loc)
@pytest.mark.skip(reason="Not sure how this actually works yet!.")
def test_arepo_ics(answer_dir,gravity,answer_store):
    import os
    num_particles = {k: 100000 for k in ["dm", "star", "gas"]}
    ics = ClusterICs("single", 1, [f"{answer_dir}/{gravity}_model_dens_tdens.h5"], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     num_particles=num_particles)
    cds.setup_arepo_ics(ics,10000,300,f"{answer_dir}/arepo_ics.h5")

