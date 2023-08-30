"""
Testing module for the ``particles`` module of ``cluster_generator``.

"""
import pytest
from numpy.random import RandomState
from cluster_generator.particles import ClusterParticles, concat_clusters
from cluster_generator.tests.utils import particle_answer_testing, \
    generate_model
from cluster_generator.utils import mylog
import numpy as np
prng = RandomState(25)
# -------------------------------------------------------------------------------------------------------------------- #
# Utility Functions ================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def plot_particles(answer_dir,parts,boxsize,name):
    """ plots the particles """
    import yt
    parts.make_radial_cut(boxsize/2)
    parts.add_offsets([boxsize/2]*3,[0,0,0])
    ds = parts.to_yt_dataset(box_size=boxsize)

    # -- producing the plot -- #
    p = yt.ParticlePlot(ds,("all","particle_position_x"),("all","particle_position_y"),("all","particle_mass"),weight_field=("all","particle_mass"))
    p.set_cmap(("all","particle_mass"),"octarine")
    p.save(f"{answer_dir}/{name}.png")
def plot_particle_velocities(answer_dir,parts,boxsize,name):
    """ plots the particles """
    import yt
    parts.make_radial_cut(boxsize/2)
    parts.add_offsets([boxsize/2]*3,[0,0,0])
    ds = parts.to_yt_dataset(box_size=boxsize)

    # -- producing the plot -- #
    p = yt.ParticlePlot(ds,("all","particle_position_x"),("all","particle_position_y"),("all","particle_velocity_x"),weight_field=("all","particle_mass"))
    p.set_cmap(("all","particle_velocity_x"),"octarine")
    p.save(f"{answer_dir}/{name}.png")

def plot_particle_energy(answer_dir,parts,boxsize,name):
    """ plots the particles """
    import yt
    parts.make_radial_cut(boxsize/2)
    parts.add_offsets([boxsize/2]*3,[0,0,0])
    ds = parts.to_yt_dataset(box_size=boxsize)

    # -- producing the plot -- #
    p = yt.ParticlePlot(ds,("gas","particle_position_x"),("gas","particle_position_y"),("gas","thermal_energy"),weight_field=("gas","particle_mass"))
    p.set_cmap(("gas","thermal_energy"),"octarine")
    p.save(f"{answer_dir}/{name}.png")
# -------------------------------------------------------------------------------------------------------------------- #
# Pytest Fixtures ==================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def particles(answer_store,answer_dir,gravity):
    import os
    if os.path.exists(f"{answer_dir}/particles_{gravity}.h5") and not answer_store:
        return ClusterParticles.from_h5_file(f"{answer_dir}/particles_{gravity}.h5")
    else:
        m = generate_model(gravity=gravity)
        dp = m.generate_dm_particles(100000, prng=prng)
        sp = m.generate_star_particles(100000, prng=prng)
        hp = m.generate_gas_particles(100000, prng=prng)
        parts = hp + dp + sp
        parts.write_particles_to_h5(f"{answer_dir}/particles_{gravity}.h5",overwrite=True)
        return parts
# -------------------------------------------------------------------------------------------------------------------- #
# Construction Tests ================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
def test_particle_construction(answer_store, answer_dir, gravity):
    """Generates particles for the given inputs and checks against previous case."""
    #  Constructing the model object
    # ---------------------------------------------------------------------------------------------------------------- #
    from copy import deepcopy
    m = generate_model(gravity)

    #  Loading particles
    # ---------------------------------------------------------------------------------------------------------------- #
    dp = m.generate_dm_particles(100000, prng=prng)
    sp = m.generate_star_particles(100000, prng=prng)
    hp = m.generate_gas_particles(100000, prng=prng)

    #  Checking the file.
    # ---------------------------------------------------------------------------------------------------------------- #
    parts = hp + dp + sp
    particle_answer_testing(parts, f"particles_{gravity}.h5", answer_store, answer_dir)
    plot_particles(answer_dir,deepcopy(parts),14000,f"particle_construction_{gravity}")

# -------------------------------------------------------------------------------------------------------------------- #
# Methods Tests ====================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def test_drop_ptype(answer_dir,answer_store,particles):
    """ testing a variety of methods regarding the particle objects."""
    from copy import deepcopy
    parts = particles

    # - testing the drop ptype - #
    parts.drop_ptypes("gas")

    assert "gas" not in parts.particle_types

def test_offsets(answer_dir,answer_store,particles):
    """ testing a variety of methods regarding the particle objects."""
    from numpy.testing import assert_allclose
    parts = particles

    # - testing the drop ptype - #
    parts.add_offsets([1000,1000,1000],[-1000,1000,1000])

    for p in parts.particle_types:
        avg_position = np.mean(parts[p,"particle_position"],axis=0)
        avg_velocity = np.mean(parts[p,"particle_velocity"],axis=0)

        assert_allclose(avg_position.d,[1000,1000,1000],rtol=0.1)
        assert_allclose(avg_velocity.d,[-1000,1000,1000],rtol=0.1)

def test_black_hole(answer_dir,answer_store,particles):
    """tests the add blackhole method"""
    parts = particles

    parts.add_black_hole(1e10)

    assert "black_hole" in parts.particle_types

# -------------------------------------------------------------------------------------------------------------------- #
# Concatenation Tests ================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
def test_concat(answer_dir,answer_store,particles,gravity):
    """tests the concatenation of particles"""
    from copy import deepcopy

    # -- loading the particles -- #
    p1,p2,p3 = deepcopy(particles),deepcopy(particles),deepcopy(particles)

    # -- loading model -- #
    m = generate_model(gravity=gravity)

    parts = concat_clusters([p1,p2,p3],[m,m,m],[[0,0,0],[2000,0,0],[3000,3000,0]],[[0,0,0],[0,0,0],[-1000,0,0]])

    plot_particles(answer_dir,deepcopy(parts),16000,"concat_test")
    plot_particle_velocities(answer_dir,deepcopy(parts),16000,"concat_test_v")
    plot_particle_energy(answer_dir,deepcopy(parts),16000,"concat_test_e")



