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
    import matplotlib.pyplot as plt
    m,d,r = generate_mdr_potential()
    pot_QUMOND = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="QUMOND")
    pot_NEWTONIAN = Potential.from_fields({"total_mass":m,"total_density":d,"radius":r},gravity="Newtonian")

    figure = plt.figure(figsize=(12,6))
    ax1,ax2 = figure.add_subplot(121),figure.add_subplot(122)

    pot_QUMOND.plot(fig=figure,ax=ax1,color="red")
    pot_NEWTONIAN.plot(fig=figure,ax=ax1,color="black")

    ax2.loglog(pot_NEWTONIAN.fields["radius"],np.gradient(pot_NEWTONIAN.fields["potential"],pot_NEWTONIAN.fields["radius"]),color="black")
    ax2.loglog(pot_QUMOND.fields["radius"],np.gradient(pot_QUMOND.fields["potential"], pot_QUMOND.fields["radius"]),color="red")
    ax2.loglog(pot_NEWTONIAN.fields["radius"],np.gradient(pot_NEWTONIAN.fields["potential"],pot_NEWTONIAN.fields["radius"])**2/(unyt_array([1.2e-10],"m/s**2").to("kpc/Myr**2").d[0]*np.ones(pot_QUMOND.fields["radius"].d.size)))
    ax2.loglog(pot_NEWTONIAN.fields["radius"],(unyt_array([1.2e-10],"m/s**2").to("kpc/Myr**2").d[0]*np.ones(pot_QUMOND.fields["radius"].d.size)))
    ax2.set_xlabel("Radius (kpc)")
    ax2.set_ylim([10e-4,10e-2])
    ax2.set_ylabel(r"$\nabla \Phi,\;\;\left[\mathrm{\frac{kpc}{Myr^2}}\right]$")
    ax2.set_title("acceleration")
    ax1.set_title("Potential")
    figure.subplots_adjust(wspace=0.4,left=0.05,right=0.95)
    figure.savefig(os.path.join(answer_dir,"pot_compare.png"))





