"""
This file contains a variety of test cases for the implemented gravity theories and their potential computations.
"""
import pytest
import numpy as np
from cluster_generator.radial_profiles import find_overdensity_radius,snfw_mass_profile,snfw_total_mass,snfw_density_profile
from unyt import unyt_array,unyt_quantity
import cluster_generator.gravity as gr
from cluster_generator.utils import G
from numpy.testing import assert_allclose
# -------------------------------------------------------------------------------------------------------------------- #
# Fixtures =========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def mdr_model():
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    m = snfw_mass_profile(M, a)

    rmin, rmax = 0.1, 2 * r200
    r = np.geomspace(rmin, rmax, 1000)
    return unyt_array(m(r), "Msun"), unyt_array(rhot(r), "Msun/kpc**3"), unyt_array(r, "kpc")

# -------------------------------------------------------------------------------------------------------------------- #
# Asymptotic Tests =================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.usefixtures("mdr_model","answer_dir")
class TestGravityAsymptotic():
    """
    These tests are asymptotic tests for the various gravity theories
    """
    low_a0 = unyt_quantity(1.2e-30,"m/s**2")

    def test_asymptotic_low_a0(self,mdr_model,answer_dir):
        from copy import deepcopy
        import matplotlib.pyplot as plt
        #  Test configuration
        # ------------------------------------------------------------------------------------------------------------ #
        _included_gravity_classes = [gr.NewtonianGravity,gr.AQUALGravity,gr.QUMONDGravity]

        #  Setup
        # ------------------------------------------------------------------------------------------------------------ #
        m,d,r = mdr_model
        potentials = []
        fields = {"total_mass":m,"total_density":d,"radius":r}
        accel = []

        #  compute
        # ------------------------------------------------------------------------------------------------------------ #
        for grav in _included_gravity_classes:
            potentials.append(grav.compute_potential(deepcopy(fields),attrs={"a_0":self.low_a0},spinner=False).d)
            accel.append(np.gradient(potentials[-1],fields["radius"].d))

        # plotting
        #--------------------------------------------------------------------------------------------------------------#
        fig,axes = plt.subplots(1,2,figsize=(10,7))

        for p in potentials:
            axes[0].loglog(fields["radius"].d,np.abs(p))
        for a in accel:
            axes[1].loglog(fields["radius"].d,np.abs(a))

        fig.savefig(f"{answer_dir}/grav_compare_asymp.png")

        for i,ac in enumerate(accel[1:],1):
            assert_allclose(accel[0],ac,rtol=1e-2,err_msg=f"gravity classes {_included_gravity_classes[0]._classname} and {_included_gravity_classes[i]._classname} were not equal to within tolerance.")

class TestNewtonianGravity():
    """
    These tests are designed to confirm that the correct results are obtained from Newtonian gravity calculations.
    """

    def test_sphere(self):
        """Test that the newtonian solver can correctly find the potential of a sphere of radius a"""
        r = unyt_array(np.geomspace(1e-1,1e4,5000),"kpc")
        d = np.zeros(r.size)
        d = unyt_array(d,"Msun/kpc**3")

        m = np.ones(r.size)
        m = unyt_array(m,"Msun")

        potential = gr.NewtonianGravity.compute_potential({"total_mass":m,"radius":r,"total_density":d},spinner=False)
        analytic_solution = lambda x: -G.d/x


        assert_allclose(analytic_solution(r.d),potential.d)

class TestAQUALGravity:
    def test_sphere(self):
        """Test that the newtonian solver can correctly find the potential of a point mass"""
        r = unyt_array(np.geomspace(1e-1,1e4,5000),"kpc")
        d = np.zeros(r.size)
        d = unyt_array(d,"Msun/kpc**3")

        m = np.ones(r.size)
        m = unyt_array(m,"Msun")

        potential = gr.AQUALGravity.compute_potential({"total_mass":m,"radius":r,"total_density":d},spinner=True)

        alpha = (G/gr.a0).to("kpc**2/Msun").d
        a_0 = gr.a0.to("kpc/Myr**2").d

        integral = lambda x: (-alpha/(2*x))*(np.sqrt(((4*x**2)/alpha) + 1)- (2*x*np.arcsinh(2*x/np.sqrt(alpha)))/(np.sqrt(alpha)) + 1)

        analytic_solution = a_0*(integral(r.d) - integral(r.d[-1]))
        assert_allclose(analytic_solution,potential.d)

class TestQUMONDGravity:
    def test_sphere(self):
        """Test that the newtonian solver can correctly find the potential of a point mass"""
        r = unyt_array(np.geomspace(1e-1,1e4,5000),"kpc")
        d = np.zeros(r.size)
        d = unyt_array(d,"Msun/kpc**3")

        m = np.ones(r.size)
        m = unyt_array(m,"Msun")

        potential = gr.QUMONDGravity.compute_potential({"total_mass":m,"radius":r,"total_density":d},spinner=True)

        alpha = (G/gr.a0).to("kpc**2/Msun").d
        a_0 = gr.a0.to("kpc/Myr**2").d

        integral = lambda x: (-alpha/(2*x))*(np.sqrt(((4*x**2)/alpha) + 1)- (2*x*np.arcsinh(2*x/np.sqrt(alpha)))/(np.sqrt(alpha)) + 1)

        analytic_solution = a_0*(integral(r.d) - integral(r.d[-1]))
        assert_allclose(analytic_solution,potential.d,rtol=1e-2)

@pytest.mark.usefixtures("mdr_model","answer_dir")
class TestEMONDGravity:
    def test_parity(self,mdr_model,answer_dir):
        """Tests that the EMOND corresponds to AQUAL when a_0(x) = a_0."""
        from copy import deepcopy
        import matplotlib.pyplot as plt
        #  Test configuration
        # ------------------------------------------------------------------------------------------------------------ #
        _included_gravity_classes = [gr.AQUALGravity,gr.EMONDGravity]

        #  Setup
        # ------------------------------------------------------------------------------------------------------------ #
        m,d,r = mdr_model
        potentials = []
        fields = {"total_mass":m,"total_density":d,"radius":r}
        accel = []
        att = [{},{"a_0":lambda x: (x*0)+gr.a0.to("kpc/Myr**2").d,"alpha":1}]
        #  compute
        # ------------------------------------------------------------------------------------------------------------ #
        for grav,ats in zip(_included_gravity_classes,att):
            potentials.append(grav.compute_potential(deepcopy(fields),attrs=ats,spinner=False).d)
            accel.append(np.gradient(potentials[-1],fields["radius"].d))

        # plotting
        #--------------------------------------------------------------------------------------------------------------#
        fig,axes = plt.subplots(1,2,figsize=(10,7))

        for p in potentials:
            axes[0].loglog(fields["radius"].d,np.abs(p))
        for a in accel:
            axes[1].loglog(fields["radius"].d,np.abs(a))

        fig.savefig(f"{answer_dir}/grav_compare_emond_equal.png")

        for i,ac in enumerate(accel[1:],1):
            assert_allclose(accel[0],ac,rtol=1e-2,err_msg=f"gravity classes {_included_gravity_classes[0]._classname} and {_included_gravity_classes[i]._classname} were not equal to within tolerance.")





