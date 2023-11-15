import os
import sys

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
    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="Incompatible dill serialization"
    )
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


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Incompatible dill serialization"
)
@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestCorrections:
    """
    Testing for the corrections methods in the models module.
    """

    mdl_name = "test_correction.h5"

    def model(self, answer_store, answer_dir):
        """Constructs the relevant model"""
        from cluster_generator.radial_profiles import (
            vikhlinin_density_profile,
            vikhlinin_temperature_profile,
        )

        if answer_store:
            if hasattr(self, "is_built") and self.is_built is True:
                # This was already built once this cycle.
                self._model = ClusterModel.from_h5_file(
                    os.path.join(answer_dir, self.mdl_name)
                )
                return self._model
            else:
                pass
        else:
            if os.path.exists(os.path.join(answer_dir, self.mdl_name)):
                self._model = ClusterModel.from_h5_file(
                    os.path.join(answer_dir, self.mdl_name)
                )
                return self._model
            else:
                pass

        density = vikhlinin_density_profile(
            119846, 94.6, 1239.9, 0.916, 0.526, 4.943, 3
        )
        temperature = vikhlinin_temperature_profile(
            3.61, 0.12, 5, 10, 1420, 0.27, 57, 3.88
        )
        self._model = ClusterModel.from_dens_and_temp(1, 10000, density, temperature)
        self._model.write_model_to_h5(
            os.path.join(answer_dir, self.mdl_name), overwrite=True
        )
        self.is_built = True
        return self._model

    def test_minimal(self, answer_store, answer_dir):
        m = self.model(answer_store, answer_dir)
        assert not m.is_physical
        new_m = m.correct()
        assert new_m.is_physical

    def test_smooth(self, answer_store, answer_dir):
        m = self.model(answer_store, answer_dir)
        assert not m.is_physical
        new_m = m.correct(mode="smooth")
        assert new_m.is_physical

    def test_compare(self, answer_store, answer_dir):
        import matplotlib.pyplot as plt

        m = self.model(answer_store, answer_dir)

        f, a = m.panel_plot()

        assert not m.is_physical

        new_m = m.correct(mode="smooth")
        new_m_minimal = self.model(answer_store, answer_dir).correct(mode="minimal")

        new_m.panel_plot(fig=f, axes=a, color="red")
        new_m_minimal.panel_plot(fig=f, axes=a, color="green")
        plt.savefig(os.path.join(answer_dir, "compare_correction.png"))

        assert new_m.is_physical
