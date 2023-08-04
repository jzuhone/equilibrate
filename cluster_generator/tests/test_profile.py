import matplotlib.pyplot as plt
import numpy as np
from cluster_generator.gravity import Potential
from cluster_generator.model import _compute_total_mass
from cluster_generator.tests.utils import model_answer_testing, \
    generate_model, generate_mdr_potential
from unyt import unyt_array
from numpy.testing import assert_allclose
def test_model(answer_store, answer_dir):
    m = generate_model()
    model_answer_testing(m, "profile.h5", answer_store, answer_dir)
    assert np.amax(m.check_hse()) < 1.0e-4

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

def test_model_temperature(answer_store,answer_dir):
    """Tests if interp_function: 1 gives the correct information."""
    attrs = {"interp_function":lambda x: 1}
    modelN = generate_model(attrs=attrs)
    modelMA = generate_model(gravity="AQUAL",attrs=attrs)
    modelMQ = generate_model(gravity="QUMOND",attrs=attrs)

    assert_allclose(modelN["temperature"],modelMA["temperature"],rtol=1e-3)
    assert_allclose(modelN["temperature"],modelMQ["temperature"],rtol=1e-3)