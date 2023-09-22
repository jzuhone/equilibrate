"""
Tests for gravity objects
"""
import pytest
import numpy as np
from cluster_generator.radial_profiles import find_overdensity_radius,snfw_mass_profile,snfw_total_mass,snfw_density_profile
from unyt import unyt_array,unyt_quantity
import cluster_generator.gravity as gr
from cluster_generator.utils import G
from numpy.testing import assert_allclose
from cluster_generator.model import ClusterModel
from cluster_generator.tests.utils import generate_model_dens_tdens

@pytest.fixture
def mdr_model():
    # This creates a model with mass-density-radius
    z = 0.1
    M200 = 1.5e15
    conc = 4.0
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    m = snfw_mass_profile(M, a)

    rmin, rmax = 0.1, 10000
    r = np.geomspace(rmin, rmax, 1000)
    return unyt_array(m(r), "Msun"), unyt_array(rhot(r), "Msun/kpc**3"), unyt_array(r, "kpc")

def standard_models_dens_tdens(answer_dir, answer_store, gravity):
    import os
    if os.path.exists(f"{answer_dir}/{gravity}_model_dens_tdens.h5") and not answer_store:
        return ClusterModel.from_h5_file(f"{answer_dir}/{gravity}_model_dens_tdens.h5")
    else:
        m = generate_model_dens_tdens(gravityity=gravity)
        m.write_model_to_h5(f"{answer_dir}/{gravity}_model_dens_tdens.h5", overwrite=True)
    return m

@pytest.mark.usefixtures("mdr_model","answer_dir")
class TestGravityAsymptotic():
    """
    These tests are asymptotic tests for the various gravity theories
    """
    low_a0 = unyt_quantity(1.2e-30,"m/s**2")

    def test_asymptotic_low_a0(self,mdr_model,answer_dir):
        from copy import deepcopy
        import matplotlib.pyplot as plt

        _included_gravity_classes = [gr.NewtonianGravity,gr.AQUALGravity,gr.QUMONDGravity]

        m,d,r = mdr_model
        potentials = []
        fields = {"total_mass":m,"total_density":d,"radius":r}
        accel = []

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
@pytest.mark.usefixtures("mdr_model","answer_dir")

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

    def test_hernquist(self):
        rr = np.geomspace(0.1,10000,10000) # the integration domain
        density_answer = lambda x: (1/(2*np.pi))*(500/x)*(1/(500+x)**3)
        mass_answer = lambda x: (x/(500+x))**2
        potential_answer = lambda x: -G/(x+500)

        test_answer = gr.NewtonianGravity.compute_potential({"total_density":unyt_array(density_answer(rr),"Msun/kpc**3"),"total_mass":unyt_array(mass_answer(rr),"Msun"),"radius":unyt_array(rr,"kpc")})

        np.testing.assert_allclose(test_answer.d,potential_answer(rr).d,rtol=0.1)
    def test_orbit(self,answer_dir):
        """Tests that the orbital computations are accurate. Checks for circular orbits."""
        from cluster_generator import orbits as orb
        import matplotlib.pyplot as plt

        # -- building the array --#
        x_0 = np.array([[0,3000],[0,0],[0,0]],dtype="float64")
        v_0 = np.array([[0,0],[0,1.223],[0,0]],dtype="float64")
        assert x_0.shape == (3,2), "The arrays aren't the right shape."

        # -- Building out the data -- #
        data = orb.newtonian_orbits(x_0,v_0,np.array([1e15,1],dtype="float64"),2,10000,1,1,10000,1,50)


        # -- Generating the figure -- #
        x_test,y_test = 3000 * np.cos(4.076e-4 * data[0]), 3000*np.sin(4.076e-4 * data[0])
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for k in range(2):
            ax.plot(data[1][0,k,:],data[1][1,k,:],alpha=0.7)

        ax.plot(x_test,y_test,color="red",ls="-.",alpha=0.4)
        ax.set_xlim([-5000,5000])
        ax.set_ylim([-5000,5000])

        ax2.semilogx(data[0],(data[1][0,1,:]-x_test)/data[1][0,1,:])
        ax2.semilogx(data[0], (data[1][1, 1, :] - y_test) / data[1][1, 1, :])
        fig.savefig(f"{answer_dir}/newtonian_orbit_circular.png")

        # -- running the first test -- #
        assert data[4] == 0, f"The orbital solver came back with error code {data[4]}."

        # -- Running the main test -- #

        r = x_test**2 + y_test**2

        np.testing.assert_allclose(r,(3000**2)*np.ones(data[0].size))

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

        analytic_solution = a_0*(integral(r.d) - integral(2*r.d[-1]))
        assert_allclose(analytic_solution,potential.d)

    @pytest.mark.usefixtures("answer_store", "answer_dir")
    def test_virialization(self,answer_dir,answer_store):
        """
        Tests the virialization of the system and its ability to create particles.
        """
        # -- Building the Model -- #
        mod = generate_model_dens_tdens(gravity="AQUAL")

        vir = mod.dm_virial

        parts = vir.generate_particles(1_000_000)

    def test_orbit(self,answer_dir):
        """Tests that the orbital computations are accurate. Checks for circular orbits."""
        from cluster_generator import orbits as orb
        import matplotlib.pyplot as plt

        # -- building the array --#
        x_0 = np.array([[0,3000],[0,0],[0,0]],dtype="float64")
        v_0 = np.array([[0,0],[0,1.223],[0,0]],dtype="float64")
        assert x_0.shape == (3,2), "The arrays aren't the right shape."

        # -- Building out the data -- #
        data = orb.aqual_orbits(x_0,v_0,np.array([1e15,1],dtype="float64"),2,10000,1,1,10000,1,50)


        # -- Generating the figure -- #
        x_test,y_test = 3000 * np.cos(6.807e-4 * data[0]), 3000*np.sin(6.807e-4 * data[0])
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        for k in range(2):
            ax.plot(data[1][0,k,:],data[1][1,k,:],alpha=0.7)

        ax.plot(x_test,y_test,color="red",ls="-.",alpha=0.4)
        ax.set_xlim([-5000,5000])
        ax.set_ylim([-5000,5000])

        ax2.semilogx(data[0],(data[1][0,1,:]-x_test)/data[1][0,1,:])
        ax2.semilogx(data[0], (data[1][1, 1, :] - y_test) / data[1][1, 1, :])
        fig.savefig(f"{answer_dir}/aqual_orbit_circular.png")

        # -- running the first test -- #
        assert data[4] == 0, f"The orbital solver came back with error code {data[4]}."

        # -- Running the main test -- #

        r = x_test**2 + y_test**2

        np.testing.assert_allclose(r,(3000**2)*np.ones(data[0].size))


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

        analytic_solution = a_0*(integral(r.d) - integral(2*r.d[-1]))
        assert_allclose(analytic_solution,potential.d,rtol=1e-2)


@pytest.mark.usefixtures("mdr_model","answer_dir")
@pytest.mark.skip(reason="Implementation not-complete.")
class TestEMONDGravity:
    def test_parity(self,mdr_model,answer_dir):
        """Tests that the EMOND corresponds to AQUAL when a_0(x) = a_0."""
        from copy import deepcopy
        import matplotlib.pyplot as plt

        _included_gravity_classes = [gr.AQUALGravity,gr.EMONDGravity]

        m,d,r = mdr_model
        potentials = []
        fields = {"total_mass":m,"total_density":d,"radius":r}
        accel = []
        att = [{},{"a_0":lambda x: (x*0)+gr.a0.to("kpc/Myr**2").d,"alpha":1}]

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





