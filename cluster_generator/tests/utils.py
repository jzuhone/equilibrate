from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_equal
from unyt import unyt_array
from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles
from cluster_generator.radial_profiles import find_overdensity_radius, \
    snfw_density_profile, snfw_total_mass, vikhlinin_density_profile, vikhlinin_temperature_profile,\
    rescale_profile_by_mass, find_radius_mass, snfw_mass_profile


def generate_model(gravity="Newtonian",attrs=None):
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)
    f_g = 0.12
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g * M500, r500)
    rhos = 0.02 * rhot
    rmin = 0.1
    rmax = 10000.0
    m = ClusterModel.from_dens_and_tden(rmin, rmax, rhog, rhot,
                                        stellar_density=rhos,gravity=gravity,attrs=attrs)
    m.set_magnetic_field_from_beta(100.0, gaussian=True)

    return m

def generate_model_dens_tdens(gravity="Newtonian",attrs=None):
    """Generates a model test for dens/tdens initialization as a model."""
    z= 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)
    f_g = 0.12
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g * M500, r500)
    rhos = 0.02 * rhot
    rmin = 0.1
    rmax = 10000.0
    m = ClusterModel.from_dens_and_tden(rmin, rmax, rhog, rhot,
                                        stellar_density=rhos,gravity=gravity,attrs=attrs)
    m.set_magnetic_field_from_beta(100.0, gaussian=True)
    return m

def generate_model_dens_temp(gravity="Newtonian",attrs=None):
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)
    f_g = 0.12
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g * M500, r500)

    temp = vikhlinin_temperature_profile(2.42,-0.02,5.00,1.1,350,1,19,2)
    rhos = 0.02 * rhot
    rmin = 0.1
    rmax = 10000.0
    m = ClusterModel.from_dens_and_temp(rmin, rmax, rhog, temp,
                                        stellar_density=rhos, gravity=gravity, attrs=attrs)
    m.set_magnetic_field_from_beta(100.0, gaussian=True)
    return m

@pytest.fixture(scope="package")
def generate_mdr_potential():
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    m = snfw_mass_profile(M,a)

    rmin,rmax = 0.1,2*r200
    r =  np.geomspace(rmin,rmax,1000)
    return unyt_array(m(r),"Msun"),unyt_array(rhot(r),"Msun/kpc**3"),unyt_array(r,"kpc")


# -------------------------------------------------------------------------------------------------------------------- #
#  Answering Protocols =============================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def model_answer_testing(model, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        model.write_model_to_h5(p, overwrite=True)
    else:
        old_model = ClusterModel.from_h5_file(p)
        for field in old_model.fields:
            assert_equal(old_model[field], model[field])
        assert_equal(old_model.dm_virial.df, model.dm_virial.df)
        assert_equal(old_model.star_virial.df, model.star_virial.df)


def particle_answer_testing(parts, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        parts.write_particles(p, overwrite=True)
    else:
        old_parts = ClusterParticles.from_file(p)
        for field in old_parts.fields:
            assert_equal(old_parts[field], parts[field])


def potential_answer_testing(potential, filename, answer_store, answer_dir):
    p = Path(answer_dir) / filename
    if answer_store:
        potential.write_potential_to_h5(p, overwrite=True)
    else:
        old_pot = potential.from_h5_file(p)
        for field in old_pot.fields:
            assert_equal(old_pot[field], potential[field])
        assert old_pot.gravity == potential.gravity
