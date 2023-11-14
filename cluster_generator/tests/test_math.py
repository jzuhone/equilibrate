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
from numpy.random import uniform
from numpy.testing import assert_allclose, assert_array_less

from cluster_generator.numeric import hfa, hia
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


@pytest.mark.usefixtures("answer_store", "answer_dir")
class TestNumerical:
    """Test instances for checking the numeric cython module"""

    @pytest.mark.parametrize("test_id", range(10))
    def test_holes(self, answer_store, answer_dir, test_id):
        """uses a cubic with a known hole spacing to check hole identification"""
        a = uniform(1, 20)

        func = lambda x: -x + (x**2 / a)

        _x = np.linspace(0, 2 * a, 1000)
        _y = func(_x)

        hi, hx, hy = hia(_x, _y)

        assert_allclose(hi, np.array([[0, 500]], dtype="int32"))
        assert_allclose(hx, np.array([[0, a]]), rtol=(2 * a) / 1000)

    def test_hfa(self, answer_store, answer_dir):
        """test hfa against profiles."""
        import os

        import matplotlib.pyplot as plt

        x = np.geomspace(0.1, 100, 1000)

        colors = [
            "#d4b954",
            "#875dd0",
            "#7dcc57",
            "#d449a3",
            "#72caab",
            "#ce6039",
            "#5e90c2",
            "#767f44",
            "#b685bd",
            "#c2636f",
        ]
        for i in range(10):
            y = (x - uniform(0.1, 10)) ** 2 * (x)

            _x, _y = np.log10(x), np.log10(y)

            u, v = hfa(_x, _y, 50)
            assert_array_less(np.zeros(v.shape) - 1e-2, v - np.maximum.accumulate(v))
            _xx, _yy = 10 ** (u), 10 ** (v)
            plt.loglog(x, y, color=colors[i])
            plt.loglog(_xx, _yy, color=colors[i], ls=":")

        plt.savefig(os.path.join(answer_dir, "mono_interp.png"))
