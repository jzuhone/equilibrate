import matplotlib.pyplot as plt
import numpy as np
import pytest

from cluster_generator.gravity import Potential
from cluster_generator.model import _compute_total_mass, ClusterModel
from cluster_generator.tests.utils import model_answer_testing, \
    generate_model, generate_mdr_potential, generate_model_dens_tdens
from unyt import unyt_array
from numpy.testing import assert_allclose
import unittest

#  Utility Functions
# ----------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def standard_models(answer_dir,answer_store):
    import os
    models = {}
    for idn,grav in enumerate(["Newtonian","QUMOND","AQUAL"]):
        if os.path.exists(f"{answer_dir}/{grav}_model.h5") and not answer_store:
            models[grav] = ClusterModel.from_h5_file(f"{answer_dir}/{grav}_model.h5")
        else:
            models[grav] = generate_model_dens_tdens(gravity=grav, attrs={})
            models[grav].write_model_to_h5(f"{answer_dir}/{grav}_model.h5",overwrite=True)
    return models

@pytest.fixture
def asymptotic_models(answer_dir):
    import os
    models = {}
    for idn,grav in enumerate(["Newtonian","QUMOND","AQUAL"]):
        if os.path.exists(f"{answer_dir}/asymptotic_{grav}_model.h5"):
            models[grav] = ClusterModel.from_h5_file(f"{answer_dir}/asymptotic_{grav}_model.h5")
        else:
            models[grav] = generate_model_dens_tdens(gravity=grav, attrs={"interp_function":lambda x: 1})
            models[grav].write_model_to_h5(f"{answer_dir}/asymptotic_{grav}_model.h5",overwrite=True)
    return models
# -------------------------------------------------------------------------------------------------------------------- #
# Tests ============================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
#  Model Tests
# ----------------------------------------------------------------------------------------------------------------- #
def test_model_generation(answer_store,answer_dir):
    """
    Tests the generation of the models and checks them against existing copies if ``answer_store`` is ``False``.
    """
    for idn,grav in enumerate(["Newtonian","QUMOND","AQUAL"]):
            model = generate_model_dens_tdens(gravity=grav,attrs={})

            model_answer_testing(model,f"{answer_dir}/{grav}_model.h5",answer_store,answer_dir)

#  Virialization Tests
# ----------------------------------------------------------------------------------------------------------------- #
def test_virialization_newtonian(answer_dir,standard_models):
    # - reading the model - #
    import os
    import matplotlib.pyplot as plt

    modelN = standard_models["Newtonian"]

    # - generating the virialization - #
    df = modelN.dm_virial
    pden = modelN[f"dark_matter_density"].d
    rho_check, chk = df.check_virial()

    fig = plt.figure()
    ax1,ax2 = fig.add_subplot(211),fig.add_subplot(212)
    ax1.loglog(modelN["radius"],pden)
    ax1.loglog(modelN["radius"],rho_check)
    ax2.loglog(modelN["radius"],np.abs(chk))

    fig.savefig(f"{answer_dir}/virial_check.png")

    assert np.mean(chk[np.where(chk != np.inf)]) < 10e-2

def test_dispersion(answer_dir,standard_models):
    """Tests the AQUAL dispersion"""
    #  Load the correct models
    # ----------------------------------------------------------------------------------------------------------------- #
    modelA = standard_models["AQUAL"]
    modelQ = standard_models["QUMOND"]
    modelN = standard_models["Newtonian"]

    modelN.virialization_method = "lma"
    #  Generate the virialization object
    # ----------------------------------------------------------------------------------------------------------------- #
    virA = modelA.dm_virial
    virQ = modelQ.dm_virial
    virN = modelN.dm_virial
    #  Generating the plot
    # ----------------------------------------------------------------------------------------------------------------- #
    import matplotlib.pyplot as plt
    plt.loglog(modelA["radius"].d,virA.sigma)
    plt.loglog(modelQ["radius"],virQ.sigma)
    plt.loglog(modelN["radius"],virN.sigma)
    plt.savefig(f"{answer_dir}/dispersion_comparison.png")

def test_total_mass(answer_store,answer_dir):
    """Tests if interp_function: 1 gives the correct information."""
    m,d,r = generate_mdr_potential()
    fields = {"total_mass":m,"radius":r,"total_density":d}


    #- testing parity -#
    potential = Potential(fields,gravity="Newtonian",attrs={"interp_function":lambda x: 1})
    fields["gravitational_field"] = -unyt_array(np.gradient(potential.pot,r.d),units="kpc/Myr**2")

    total_mass_N = _compute_total_mass(fields,gravity="Newtonian",attrs={"interp_function":lambda x: 1})
    total_mass_MA = _compute_total_mass(fields,gravity="AQUAL",attrs={"interp_function":lambda x: 1})
    total_mass_MQ = _compute_total_mass(fields,gravity="QUMOND",attrs={"interp_function":lambda x: 1})


    assert_allclose(total_mass_N.d,total_mass_MQ.d,rtol=1e-3)
    assert_allclose(total_mass_N.d,total_mass_MA.d,rtol=1e-3)

def test_model_temperature(answer_store,answer_dir,asymptotic_models):
    """Tests if interp_function: 1 gives the correct information."""
    modelN,modelMA,modelMQ = asymptotic_models["Newtonian"],asymptotic_models["AQUAL"],asymptotic_models["QUMOND"]

    assert_allclose(modelN["temperature"],modelMA["temperature"],rtol=1e-3)
    assert_allclose(modelN["temperature"],modelMQ["temperature"],rtol=1e-3)

