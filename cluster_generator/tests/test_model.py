import os

import numpy as np
import pytest
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.tests.utils import generate_model, model_answer_testing


def playmodel(answer_dir):
    """
    Produces a mock model that can be used for rapid property testing. Always fetches old if one exists.
    """
    if os.path.exists(os.path.join(answer_dir, "profile.h5")):
        return ClusterModel.from_h5_file(os.path.join(answer_dir, "profile.h5"))
    else:
        m = generate_model()
        m.write_model_to_h5(os.path.join(answer_dir, "profile.h5"))
        return m


@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestModels:
    """
    Tests specific to the model.ClusterModel class and its methods.
    """

    def test_model(self, answer_store, answer_dir):
        """
        Core test for model generation. Checks that the generated model matches previous and successfully generates.

        !! Note: Generates the profile.h5 file which is used in test_ics.py.
        """
        m = generate_model()
        model_answer_testing(m, "profile.h5", answer_store, answer_dir)
        assert np.all(m.check_hse() < 1.0e-4)

    @pytest.mark.noncritical
    def test_properties(self, answer_dir):
        """Calls the properties of the current model, saves the model to file, reads it and checks that they are the same."""
        model = playmodel(answer_dir)

        properties = model.properties
        model.write_model_to_h5(os.path.join(answer_dir, "tmp_test_properties.h5"))
        new_model = ClusterModel.from_h5_file(
            os.path.join(answer_dir, "tmp_test_properties.h5")
        )

        os.remove(os.path.join(answer_dir, "tmp_test_properties.h5"))
        os.remove(os.path.join(answer_dir, "tmp_test_properties.h5.properties"))

        assert new_model.properties == properties

    @pytest.mark.noncritical
    def test_from_arrays(self):
        """Confirms that there are no dangling errors when initializing from fields."""
        r = unyt_array(np.linspace(1, 5000, 5000), "kpc")
        dens = unyt_array(5000 * r.d ** (-2), "Msun/kpc**3")

        m = ClusterModel.from_arrays({"radius": r, "density": dens})

        assert m.properties["meth"]["method"] == "from_arrays"
        assert np.array_equal(m["radius"].d, r)
        assert np.array_equal(dens, m["density"].d)

    @pytest.mark.skip("Test Not Yet Written")
    def test_model_mag_field(self):
        pass

    @pytest.mark.skip("Test Not Yet Written")
    def test_tracers(self):
        pass

    @pytest.mark.skip("Test Not Yet Written")
    def test_dataset(self):
        pass
