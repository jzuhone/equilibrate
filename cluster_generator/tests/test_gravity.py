import numpy as np

from cluster_generator.tests.utils import model_answer_testing, \
    generate_model, generate_mdr_potential, potential_answer_testing
from numpy.testing import assert_allclose
from cluster_generator.gravity import Potential
from cluster_generator.utils import cg_params
import os
from unyt import unyt_array, unyt_quantity
from scipy.interpolate import InterpolatedUnivariateSpline

# -------------------------------------------------------------------------------------------------------------------- #
# Utilities ========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
colors = {"AQUAL":"red","QUMOND":"blue","Newtonian":"black"}
lss = {"AQUAL":":","QUMOND":"--","Newtonian":"-."}
def plot_accelerations(potentials,output_path,title):
    # Plots the acceleration comparisons and returns the acceleration arrays #
    import matplotlib.pyplot as plt

    #- setting up the figure -#
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #- plotting -#
    out = []
    for k,v in potentials.items():
        ax.loglog(v["radius"],np.abs(np.gradient(v.pot.to("m**2/s**2"),v["radius"].to("m"))),color=colors[k],label=k,ls=lss[k],alpha=0.75)
        out.append(np.abs(np.gradient(v.pot.to("m**2/s**2"),v["radius"].to("m"))))

    ax.hlines(xmin=np.amin(v["radius"].d),xmax=np.amax(v["radius"].d),y=cg_params["mond","a_0"].to("m/s**2").d,label=r"$a_0$")

    ax.loglog(potentials["Newtonian"]["radius"].d,np.sqrt(cg_params["mond","a_0"].to("m/s**2").d*np.abs(np.gradient(potentials["Newtonian"].pot.to("m**2/s**2"),potentials["Newtonian"]["radius"].to("m")))),ls=":",c="magenta",alpha=0.5,label=r"$\sqrt{g_N a_0}$")
    ax.legend()
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Acceleration [m/s^2]")
    ax.set_title(title)
    plt.savefig(output_path)

    return out

# -------------------------------------------------------------------------------------------------------------------- #
# Base Change Tests ================================================================================================== #
# ------------------                                                                                                   #
# These tests check whether recent updates to the code change the physics                                              #
# -------------------------------------------------------------------------------------------------------------------- #
def test_potential_mdr_newtonian(answer_store,answer_dir,generate_mdr_potential):
    #- Setting up -#
    m,d,r = generate_mdr_potential

    #- generating the potential -#
    pot = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="Newtonian")
    f,a = pot.plot()
    f.savefig(os.path.join(answer_dir,"pot_mdr_newt.png"))

    potential_answer_testing(pot,"pot_mdr_newt.h5",answer_store,answer_dir)

def test_potential_mdr_qumond(answer_store,answer_dir,generate_mdr_potential):
    #- Setting up -#
    m,d,r = generate_mdr_potential

    #- generating the potential -#
    pot = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="QUMOND")
    f,a = pot.plot()
    f.savefig(os.path.join(answer_dir,"pot_mdr_qumond.png"))

    potential_answer_testing(pot,"pot_mdr_qumond.h5",answer_store,answer_dir)

def test_potential_mdr_aqual(answer_store,answer_dir,generate_mdr_potential):
    #- Setting up -#
    m,d,r = generate_mdr_potential

    #- generating the potential -#
    pot = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="AQUAL")
    f,a = pot.plot()
    a.loglog(pot["radius"],np.abs(pot["guess_potential"]))
    f.savefig(os.path.join(answer_dir,"pot_mdr_aqual.png"))


    potential_answer_testing(pot,"pot_mdr_aqual.h5",answer_store,answer_dir)

# -------------------------------------------------------------------------------------------------------------------- #
# Asymptotic Tests =================================================================================================== #
#------------------                                                                                                    #
# These tests check if the MOND behaviors match the correct asymptotes                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
def test_accel_low_a0(answer_dir,answer_store,generate_mdr_potential):
    # -- Setup -- #
    m,d,r = generate_mdr_potential

    # -- Getting the potentials -- #
    cg_params["mond","interp_alpha"] = 1
    cg_params["mond","a_0"] = unyt_quantity(1.2e-15,"m/s**2")

    # -- producing the potentials -- #
    pa,pn,pq = [Potential.from_fields({"total_mass": m, "total_density": d, "radius": r},gravity=grav) for grav in ["AQUAL","Newtonian","QUMOND"]]

    # -- plotting -- #
    q = plot_accelerations({
        g:i for g,i in zip(["AQUAL","Newtonian","QUMOND"],[pa,pn,pq])
    }, output_path=f"{answer_dir}/accel_low_a0.png",title=r"Low $a_0$ comparison")

    # -- asserting similar -- #
    assert_allclose(q[0],q[1],rtol=1e-3,verbose=True)
    assert_allclose(q[0],q[2],rtol=1e-3,verbose=True)

def test_accel_sharp_interp(answer_dir,answer_store,generate_mdr_potential):
    # -- Setup -- #
    m,d,r = generate_mdr_potential

    # -- Getting the potentials -- #
    cg_params["mond", "a_0"] = unyt_quantity(1.2e-10, "m/s**2")
    cg_params["mond","interp_alpha"] = 100

    # -- producing the potentials -- #
    pa,pn,pq = [Potential.from_fields({"total_mass": m, "total_density": d, "radius": r},gravity=grav) for grav in ["AQUAL","Newtonian","QUMOND"]]

    # -- plotting -- #
    q = plot_accelerations({
        g:i for g,i in zip(["AQUAL","Newtonian","QUMOND"],[pa,pn,pq])
    }, output_path=f"{answer_dir}/accel_high_alpha.png",title=r"Rapid interpolation comparison")

    # -- comparing -- #
    q = [i[np.where(i >= cg_params["mond","a_0"].to("m/s**2").d)] for i in q]
    print([len(i) for i in q])
    ml = np.amin(np.array([len(i) for i in q]))

    #TODO: this is kind of a weak tolerance
    assert_allclose(q[0][:ml-1],q[1][:ml-1],rtol=1)
    assert_allclose(q[0][:ml-1],q[1][:ml-1],rtol=1)

def test_accel_far_field(answer_dir,answer_store,generate_mdr_potential):
    # -- Setup -- #
    m,d,r = generate_mdr_potential

    # -- Getting the potentials -- #
    cg_params["mond", "a_0"] = unyt_quantity(1.2e-10, "m/s**2")
    cg_params["mond","interp_alpha"] = 1

    # -- producing the potentials -- #
    pa,pn,pq = [Potential.from_fields({"total_mass": m, "total_density": d, "radius": r},gravity=grav) for grav in ["AQUAL","Newtonian","QUMOND"]]

    # -- plotting -- #
    q = plot_accelerations({
        g:i for g,i in zip(["AQUAL","Newtonian","QUMOND"],[pa,pn,pq])
    }, output_path=f"{answer_dir}/accel_far_field.png",title=r"Rapid far_field check")



def test_diff_mdr(answer_dir,generate_mdr_potential):
    import matplotlib.pyplot as plt

    m, d, r = generate_mdr_potential

    #- Generating the potential class objects from fields -#
    pot_AQUAL = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="AQUAL")
    pot_NEWTONIAN = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="Newtonian")
    pot_QUMOND = Potential.from_fields({"total_mass": m, "total_density": d, "radius": r}, gravity="QUMOND")

    #- Plotting
    figure = plt.figure(figsize=(8,5))
    ax = figure.add_subplot(121)
    ax2 = figure.add_subplot(122)
    pot_AQUAL.plot(fig=figure,ax=ax,color="b",label="AQUAL")
    pot_NEWTONIAN.plot(fig=figure,ax=ax,color="k",label="Newtonian")
    pot_QUMOND.plot(fig=figure,ax=ax,color="r",label="QUMOND")
    ax.legend()
    ax2.loglog(pot_QUMOND["radius"].d,np.gradient(pot_QUMOND["gravitational_potential"].d,pot_QUMOND["radius"].d),"r")
    ax2.loglog(pot_AQUAL["radius"].d,np.gradient(pot_AQUAL["gravitational_potential"].d,pot_AQUAL["radius"].d),"b")
    ax2.loglog(pot_NEWTONIAN["radius"].d,np.gradient(pot_NEWTONIAN["gravitational_potential"].d,pot_NEWTONIAN["radius"].d),"k")
    ax2.hlines(y=unyt_array(1.2e-10,"m/s**2").to("kpc/Myr**2").d,xmin=np.amin(r.d),xmax=np.amax(r.d),color="c",ls=":")
    ax2.loglog(pot_NEWTONIAN["radius"].d,(np.gradient(pot_NEWTONIAN["gravitational_potential"].d,pot_NEWTONIAN["radius"].d)**2)/unyt_array(1.2e-10,"m/s**2").to("kpc/Myr**2").d,"r:")
    ax2.loglog(pot_NEWTONIAN["radius"].d,(np.gradient(pot_NEWTONIAN["gravitational_potential"].d,pot_NEWTONIAN["radius"].d)*unyt_array(1.2e-10,"m/s**2").to("kpc/Myr**2").d)**(1/2),"b:")
    ax2.set_xlabel("Radius [kpc]")
    ax2.set_ylabel(r"$\left|a\right|,\;\;\left[\mathrm{\frac{kpc}{Myr^2}}\right]$")
    ax.set_ylabel(r"$\left|\Phi\right|,\;\;\left[\mathrm{\frac{kpc^2}{Myr^2}}\right]$")
    ax2.text(0.17,6e-2,r"$\frac{a_{\mathrm{newt}^2}}{a_0}$")
    ax2.text(0.17,1.4e-2,r"$a_{\mathrm{newt}}$")
    ax2.text(0.17,4e-3,r"$\sqrt{a_0a_{\mathrm{newt}}}$")
    ax.set_title("Potential")
    ax2.set_title("Acceleration")
    plt.subplots_adjust(wspace=0.3)
    figure.savefig(os.path.join(answer_dir,"potential_comp.png"))






