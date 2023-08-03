import numpy as np

from cluster_generator.tests.utils import model_answer_testing, \
    generate_model, generate_mdr_potential, potential_answer_testing
from cluster_generator.gravity import Potential
import os
from unyt import unyt_array

def test_potential_mdr_newtonian(answer_store,answer_dir):
    #- Setting up -#
    m,d,r = generate_mdr_potential()

    #- generating the potential -#
    pot = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="Newtonian")
    f,a = pot.plot()
    f.savefig(os.path.join(answer_dir,"pot_mdr_newt.png"))

    potential_answer_testing(pot,"pot_mdr_newt.h5",answer_store,answer_dir)

def test_potential_mdr_qumond(answer_store,answer_dir):
    #- Setting up -#
    m,d,r = generate_mdr_potential()

    #- generating the potential -#
    pot = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="QUMOND")
    f,a = pot.plot()
    f.savefig(os.path.join(answer_dir,"pot_mdr_qumond.png"))

    potential_answer_testing(pot,"pot_mdr_qumond.h5",answer_store,answer_dir)

def test_potential_mdr_aqual(answer_store,answer_dir):
    #- Setting up -#
    m,d,r = generate_mdr_potential()

    #- generating the potential -#
    pot = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="AQUAL")
    f,a = pot.plot()
    a.loglog(pot["radius"],np.abs(pot["guess_potential"]))
    f.savefig(os.path.join(answer_dir,"pot_mdr_aqual.png"))


    potential_answer_testing(pot,"pot_mdr_aqual.h5",answer_store,answer_dir)

def test_diff_mdr(answer_dir):
    from cluster_generator.tests.utils import generate_mdr_potential
    import matplotlib.pyplot as plt

    m, d, r = generate_mdr_potential() #: Generating the profile from an SNFW profile

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
    ax2.loglog(pot_QUMOND["radius"].d,np.gradient(pot_QUMOND["potential"].d,pot_QUMOND["radius"].d),"r")
    ax2.loglog(pot_AQUAL["radius"].d,np.gradient(pot_AQUAL["potential"].d,pot_AQUAL["radius"].d),"b")
    ax2.loglog(pot_NEWTONIAN["radius"].d,np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d),"k")
    ax2.hlines(y=unyt_array(1.2e-10,"m/s**2").to("kpc/Myr**2").d,xmin=np.amin(r.d),xmax=np.amax(r.d),color="c",ls=":")
    ax2.loglog(pot_NEWTONIAN["radius"].d,(np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d)**2)/unyt_array(1.2e-10,"m/s**2").to("kpc/Myr**2").d,"r:")
    ax2.loglog(pot_NEWTONIAN["radius"].d,(np.gradient(pot_NEWTONIAN["potential"].d,pot_NEWTONIAN["radius"].d)*unyt_array(1.2e-10,"m/s**2").to("kpc/Myr**2").d)**(1/2),"b:")
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






