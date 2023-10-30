"""
Test file containing mathematics tests for the various minor algorithms in this code base.

=====
Tests
=====

TestIntegration
---------------
    - test_integrate_mass: Checks that the integrate_mass function performs as expected.

"""
import numpy as np
import pytest

from cluster_generator.utils import integrate_mass


@pytest.mark.noncritical
class TestIntegration:
    """This class is used to stress test the numerical methods used here."""

    def test_integrate_mass(self):
        rr = np.geomspace(0.1, 10000, 1000)  # the integration domain
        profile = lambda x: (1 / (2 * np.pi)) * (500 / x) * (1 / (500 + x) ** 3)
        answer_profile = lambda x: (x / (500 + x)) ** 2

        int = integrate_mass(profile, rr)

        np.testing.assert_allclose(int, answer_profile(rr))


@pytest.mark.noncritical
class TestInterpolation:
    """This set of tests is designed to check that custom interpolation methods are functioning as desired."""
