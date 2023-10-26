"""
This file tests the correction.py module
"""
import os
import sys

import numpy as np
import pytest

from cluster_generator.correction import (
    NonPhysicalRegion,
    Type0aNPR,
    Type0bNPR,
    Type1aNPR,
)
from cluster_generator.model import ClusterModel
from cluster_generator.radial_profiles import (
    RadialProfile,
    hernquist_density_profile,
    vikhlinin_density_profile,
    vikhlinin_temperature_profile,
)


@pytest.mark.usefixtures("answer_store", "answer_dir")
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Incompatible dill serialization"
)
class TestNPR:
    """
    This is just a template class for a correction type
    """

    mdl_name = "test_correction_generic.h5"
    npr = NonPhysicalRegion

    def model(self, answer_store, answer_dir):
        return None

    def _answers(self):
        """Stores the answer ranges to check against."""
        return {}

    def test_identification(self, answer_store, answer_dir):
        if self.model(answer_store, answer_dir) is None:
            # --> This is just the generic instance.
            return None
        else:
            # --> Testing is actually going to occur.
            u = self.npr.identify(self.model(answer_store, answer_dir), recursive=False)
            print(u)
            for k in self._answers():
                assert any(
                    j.is_close(k) for j in u
                ), f"No NPR close to {k} in findings."

    def test_correction(self, answer_dir, answer_store):
        """Tests the capacity of the instances to correct the non-physicalities"""
        if self.npr.correctable is False:
            return True

        m = self.model(answer_store, answer_dir)

        print(m)
        f, a = m.panel_plot(color="red")
        m = NonPhysicalRegion.correct(m, recursive=True)

        f, a = m.panel_plot(color="blue", fig=f, axes=a)

        f.savefig(f"{answer_dir}/{self.mdl_name}_corrected.png")
        u = self.npr.identify(m, recursive=False)

        assert len(u) == 0, "Failed to solve all of the non-physicalities of this type."


@pytest.mark.usefixtures("answer_store", "answer_dir")
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Incompatible dill serialization"
)
class TestNPR0a(TestNPR):
    """Tests NPR0a Problems"""

    mdl_name = "test_correction_NRP0a.h5"
    npr = Type0aNPR

    def model(self, answer_store, answer_dir):
        if answer_store:
            print(answer_store, hasattr(self, "is_built"))
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

        density = hernquist_density_profile(1e14, 500)
        temperature = RadialProfile(
            lambda x: 20 * np.exp(-1 * np.cosh(np.log(x / 5) / 2))
            - 20 * np.exp(-((np.log(x / 10)) ** 2))
        )
        self._model = ClusterModel.from_dens_and_temp(1, 1000, density, temperature)
        self._model.write_model_to_h5(
            os.path.join(answer_dir, self.mdl_name), overwrite=True
        )
        self.is_built = True

        return self._model

    def _answers(self):
        return [Type0aNPR(3.644, 34.00, object)]


@pytest.mark.usefixtures("answer_store", "answer_dir")
@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Incompatible dill serialization"
)
class TestNPR0b(TestNPR):
    """Tests NPR0a Problems"""

    mdl_name = "test_correction_NRP0b.h5"
    npr = Type0bNPR

    def model(self, answer_store, answer_dir):
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

        density = hernquist_density_profile(-1e14, 500)
        temperature = RadialProfile(
            lambda x: 20 * np.exp(-1 * np.cosh(np.log(x / 5) / 2))
        )
        self._model = ClusterModel.from_dens_and_temp(1, 1000, density, temperature)
        self._model.write_model_to_h5(
            os.path.join(answer_dir, self.mdl_name), overwrite=True
        )
        self.is_built = True
        return self._model

    def _answers(self):
        return [Type0bNPR(1, 1000, object)]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Incompatible dill serialization"
)
class TestNPR1a(TestNPR):
    mdl_name = "test_correction_NRP1a.h5"
    npr = Type1aNPR

    def model(self, answer_store, answer_dir):
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
        self._model = ClusterModel.from_dens_and_temp(1, 1000, density, temperature)
        self._model.write_model_to_h5(
            os.path.join(answer_dir, self.mdl_name), overwrite=True
        )
        self.is_built = True
        return self._model

    def _answers(self):
        return [Type1aNPR(1.956e1, 7.275e1, object)]
