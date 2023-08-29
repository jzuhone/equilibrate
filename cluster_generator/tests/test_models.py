import pytest

from cluster_generator.model import ClusterModel
from cluster_generator.tests.utils import generate_model_dens_tdens, generate_model_dens_temp, model_answer_testing


# -------------------------------------------------------------------------------------------------------------------- #
# Fixtures =========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def standard_models_dens_temp(answer_dir, answer_store, gravity):
    import os
    if os.path.exists(f"{answer_dir}/{gravity}_model_dens_temp.h5") and not answer_store:
        return ClusterModel.from_h5_file(f"{answer_dir}/{gravity}_model_dens_temp.h5")
    else:
        m = generate_model_dens_temp(gravity=gravity)
        m.write_model_to_h5(f"{answer_dir}/{gravity}_model_dens_temp.h5", overwrite=True)
        return m


@pytest.fixture
def standard_models_dens_tdens(answer_dir, answer_store, gravity):
    import os
    if os.path.exists(f"{answer_dir}/{gravity}_model_dens_tdens.h5") and not answer_store:
        return ClusterModel.from_h5_file(f"{answer_dir}/{gravity}_model_dens_tdens.h5")
    else:
        m = generate_model_dens_tdens(gravityity=gravity)
        m.write_model_to_h5(f"{answer_dir}/{gravity}_model_dens_tdens.h5", overwrite=True)
    return m


# -------------------------------------------------------------------------------------------------------------------- #
# Construction Tests ================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
def test_model_generation_dens_tdens(answer_store, gravity, answer_dir):
    model = generate_model_dens_tdens(gravity=gravity, attrs={})
    model_answer_testing(model, f"{answer_dir}/{gravity}_model.h5", answer_store, answer_dir)


def test_model_generation_dens_temp(answer_store, gravity, answer_dir):
    model = generate_model_dens_temp(gravity=gravity, attrs={})
    model_answer_testing(model, f"{answer_dir}/{gravity}_model.h5", answer_store, answer_dir)
