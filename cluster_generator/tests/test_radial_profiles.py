"""
Tests for the ``RadialProfile`` objects
"""
import logging
import os
import pathlib as pt

import pytest
from numpy.testing import assert_array_equal

from cluster_generator.radial_profiles import *

# -------------------------------------------------------------------------------------------------------------------- #
# Input Data ========================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
_params = {  # Stores all of the parameters for the generation of each of the test cases.
    "constant_profile"               : [1],
    "power_law_profile"              : [1, 1000, -2],
    "beta_model_profile"             : [1, 1000, 2],
    "hernquist_density_profile"      : [1, 1000],
    "cored_hernquist_density_profile": [1, 1000, 20],
    "hernquist_mass_profile"         : [1, 1000],
    "nfw_density_profile"            : [1, 1000],
    "nfw_mass_profile"               : [1, 1000],
    "tnfw_density_profile"           : [1, 1000, 1200],
    "tnfw_mass_profile"              : [1, 1000, 1200],
    "snfw_density_profile"           : [1, 1000],
    "snfw_mass_profile"              : [1, 1000],
    "einasto_density_profile"        : [1, 1000, 2],
    "einasto_mass_profile"           : [1, 1000, 2],
    "am06_density_profile"           : [1, 0.5, 0.7, 50, 2],
    "vikhlinin_density_profile"      : [1, 0.75, 0.3, 0.6, -3, 4],
    "vikhlinin_temperature_profile"  : [1, 0.1, 0.5, 1, 500, 1, 20, 2],
    "am06_temperature_profile"       : [1, 0.1, 0.3, 50]
}


@pytest.mark.filterwarnings("ignore:Casting")
def test_profiles(answer_dir, answer_store):
    """Tests for consistency"""

    # -- checking if we are writing -- #
    output_directory = f"{answer_dir}/pkl"
    pt.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # -- building and checking -- #
    x = np.geomspace(0.1, 1e6, 10000)

    for name, args in _params.items():
        try:
            _f = globals()[name](*args)
        except KeyError:
            raise ValueError(f"Failed to find profile {name} in globals. Did you write the name correctly?")

        if answer_store:
            _f.to_binary(os.path.join(output_directory, f"{name}.rp"))
        else:
            # -- actually checking -- #
            try:
                old = RadialProfile.from_binary(os.path.join(output_directory, f"{name}.rp"))
            except FileNotFoundError:
                logging.info(f"The profile {name} did not have a prior instance in the pkl file.")
                _f.to_binary(os.path.join(output_directory, f"{name}.rp"))
                continue

            assert_array_equal(old(x), _f(x), err_msg=f"Failed to match prior values for {name}.")
