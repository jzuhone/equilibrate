"""
Tests for the ``RadialProfile`` objects
"""

import os
import pathlib as pt

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import cluster_generator.radial_profiles as rp
from cluster_generator.utils import integrate_mass

_params = (
    {  # Stores all of the parameters for the generation of each of the test cases.
        "constant_profile": [1],
        "power_law_profile": [1, 1000, -2],
        "beta_model_profile": [1, 1000, 2],
        "hernquist_density_profile": [1, 1000],
        "cored_hernquist_density_profile": [1, 1000, 20],
        "hernquist_mass_profile": [1, 1000],
        "nfw_density_profile": [1, 1000],
        "nfw_mass_profile": [1, 1000],
        "tnfw_density_profile": [1, 1000, 1200],
        "tnfw_mass_profile": [1, 1000, 1200],
        "snfw_density_profile": [1, 1000],
        "snfw_mass_profile": [1, 1000],
        "cored_snfw_density_profile": [1, 1000, 600],
        "cored_snfw_mass_profile": [1, 1000, 600],
        "einasto_density_profile": [1, 1000, 2],
        "einasto_mass_profile": [1, 1000, 2],
        "am06_density_profile": [1, 0.5, 0.7, 50, 2],
        "vikhlinin_density_profile": [1, 0.75, 0.3, 0.6, -3, 4],
        "vikhlinin_temperature_profile": [1, 0.1, 0.5, 1, 500, 1, 20, 2],
        "am06_temperature_profile": [1, 0.1, 0.3, 50],
        "ad07_density_profile": [1, 1, 1, 1, 1],
        "ad07_temperature_profile": [1, 1, 1, 1],
        "baseline_entropy_profile": [1, 1, 1, 1],
        "broken_entropy_profile": [1, 1, 1],
        "walker_entropy_profile": [1, 1, 1, 1],
    }
)


@pytest.mark.parametrize("profile", list(rp.DEFAULT_PROFILE_REGISTRY.keys()))
def test_profiles(profile: str, answer_dir: str, answer_store: bool, temp_dir: str):
    """
    Test that radial profiles successfully produce arrays and match answers.
    """
    # Setup
    _r = np.geomspace(1, 10000, 1000)
    _answer_file = pt.Path(os.path.join(answer_dir, "radial_profiles.h5"))

    # Ensure that we have a parameter set.
    assert (
        profile in _params
    ), f"The profile {profile} is not present in the testing parameters."

    # Generate the profile
    radial_profile = getattr(rp.DEFAULT_PROFILE_REGISTRY, profile)(*_params[profile])
    output = radial_profile(_r)

    # Checking answers
    with h5py.File(_answer_file, "a") as fio:
        if answer_store:
            if profile in fio.keys():
                del fio[profile]

            fio.create_dataset(profile, data=output)

        else:
            check_output = fio[profile][:]
            assert_allclose(output, check_output, rtol=1e-7)


class TestUtilities:
    """This class is used to stress test the numerical nethods used here."""

    def test_integrate_mass(self):
        rr = np.geomspace(0.1, 10000, 1000)  # the integration domain
        profile = lambda x: (1 / (2 * np.pi)) * (500 / x) * (1 / (500 + x) ** 3)
        answer_profile = lambda x: (x / (500 + x)) ** 2

        int = integrate_mass(profile, rr)

        np.testing.assert_allclose(int, answer_profile(rr))
