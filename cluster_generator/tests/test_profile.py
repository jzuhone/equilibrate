import matplotlib.pyplot as plt
import numpy as np
import pytest
import yt
from numpy.testing import assert_allclose
from unyt import unyt_array

from cluster_generator.gravity import Potential
from cluster_generator.model import _compute_total_mass, ClusterModel
from cluster_generator.tests.utils import model_answer_testing, \
    generate_mdr_potential, generate_model_dens_tdens, generate_model_dens_temp

# -------------------------------------------------------------------------------------------------------------------- #
#  Constants ========================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
colors = {"AQUAL": "red", "QUMOND": "blue", "Newtonian": "black"}
lss = {"AQUAL": ":", "QUMOND": "--", "Newtonian": "-."}


# -------------------------------------------------------------------------------------------------------------------- #
# Utility Functions ================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def standard_models_dens_tdens(answer_dir, answer_store):
    import os
    models = {}
    for idn, grav in enumerate(["Newtonian", "QUMOND", "AQUAL"]):
        if os.path.exists(f"{answer_dir}/{grav}_model_dens_tdens.h5") and not answer_store:
            models[grav] = ClusterModel.from_h5_file(f"{answer_dir}/{grav}_model_dens_tdens.h5")
        else:
            models[grav] = generate_model_dens_tdens(gravity=grav)
            models[grav].write_model_to_h5(f"{answer_dir}/{grav}_model_dens_tdens.h5", overwrite=True)
    return models


@pytest.fixture
def asymptotic_models_dens_tdens(answer_dir, answer_store):
    import os
    models = {}
    for idn, grav in enumerate(["Newtonian", "QUMOND", "AQUAL"]):
        if os.path.exists(f"{answer_dir}/asymptotic_{grav}_model_dens_tdens.h5") and not answer_store:
            models[grav] = ClusterModel.from_h5_file(f"{answer_dir}/asymptotic_{grav}_model_dens_tdens.h5")
        else:
            models[grav] = generate_model_dens_tdens(gravity=grav, interp_function= lambda x: 1)
            models[grav].write_model_to_h5(f"{answer_dir}/asymptotic_{grav}_model_dens_tdens.h5", overwrite=True)
    return models


@pytest.fixture
def standard_models_dens_temp(answer_dir, answer_store):
    import os
    models = {}
    for idn, grav in enumerate(["Newtonian", "QUMOND", "AQUAL"]):
        if os.path.exists(f"{answer_dir}/{grav}_model_dens_temp.h5") and not answer_store:
            models[grav] = ClusterModel.from_h5_file(f"{answer_dir}/{grav}_model_dens_temp.h5")
        else:
            models[grav] = generate_model_dens_temp(gravity=grav)
            models[grav].write_model_to_h5(f"{answer_dir}/{grav}_model_dens_temp.h5", overwrite=True)
    return models


@pytest.fixture
def asymptotic_models_dens_temp(answer_dir, answer_store):
    import os
    models = {}
    for idn, grav in enumerate(["Newtonian", "QUMOND", "AQUAL"]):
        if os.path.exists(f"{answer_dir}/asymptotic_{grav}_model_dens_temp.h5") and not answer_store:
            models[grav] = ClusterModel.from_h5_file(f"{answer_dir}/asymptotic_{grav}_model_dens_temp.h5")
        else:
            models[grav] = generate_model_dens_temp(gravity=grav, interp_function= lambda x: 1)
            models[grav].write_model_to_h5(f"{answer_dir}/asymptotic_{grav}_model_dens_temp.h5", overwrite=True)
    return models


def plot_models(models, fields, output, diff=False):
    print(fields.shape)
    # Plots the models and fields #
    rows, cols = fields.shape[0] * (2 if diff else 1), fields.shape[1]
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, gridspec_kw={"hspace": 0, "wspace": 0.3},
                             height_ratios=([1] * rows if not diff else [1, 0.1] * int(rows / 2)),
                             figsize=(5 * cols, 3 * rows))

    raxes = axes if not diff else axes[::2, :]
    out = {}
    for i, d in enumerate(zip(raxes.ravel(), fields.ravel())):
        ax, f = d
        out[f] = []
        for k, model in models.items():
            ax.loglog(model["radius"].d, np.abs(model[f].d), c=colors[model.gravity])
            out[f].append(model[f].d)

        if diff:
            for k, model in models.items():
                axes[1::2, :].ravel()[i].semilogx(model["radius"],
                                                  np.abs(model[f].d - models["Newtonian"][f].d) / models["Newtonian"][
                                                      f].d, c=colors[model.gravity])

            axes[1::2, :].ravel()[i].set_ylim([-1e1, 1e1])
        ax.set_yscale("log")
        ax.set_ylabel(f)
    fig.savefig(output)
    return out


# -------------------------------------------------------------------------------------------------------------------- #
# Tests ============================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
# Consistency Testing ================================================================================================ #
# -------------------                                                                                                  #
# These tests are intended to test for changes in the underlying properties of models between updates.                 #
# -------------------------------------------------------------------------------------------------------------------- #
def test_model_generation_dens_tdens(answer_store, answer_dir):
    """
    Tests the generation of the models and checks them against existing copies if ``answer_store`` is ``False``.
    """
    for idn, grav in enumerate(["Newtonian", "QUMOND", "AQUAL"]):
        model = generate_model_dens_tdens(gravity=grav, attrs={})
        model_answer_testing(model, f"{answer_dir}/{grav}_model.h5", answer_store, answer_dir)


def test_model_generation_dens_temp(answer_store, answer_dir):
    """
    Tests the generation of the models and checks them against existing copies if ``answer_store`` is ``False``.
    """
    for idn, grav in enumerate(["Newtonian", "QUMOND", "AQUAL"]):
        model = generate_model_dens_temp(gravity=grav, attrs={},require_physical="rebuild")
        model_answer_testing(model, f"{answer_dir}/{grav}_model.h5", answer_store, answer_dir)

def test_rebuild(answer_store,answer_dir):
    """Test that the rebuilding system actually works"""
    models = [generate_model_dens_temp(require_physical=i) for i in [True,False,"rebuild"]]

    figure, axes = plt.subplots(6,4,gridspec_kw={"hspace":0,"wspace":0.3},height_ratios=[1,0.1,1,0.1,1,0.1],figsize=(20,20))
    plt.subplots_adjust(left=0.1,right=.99,bottom=0.1,top=.99)
    colors = ["forestgreen","black","magenta"]
    ls = ["-.",":","-"]
    fields = ["gas_mass","dark_matter_mass","stellar_mass","total_mass",
              "density","dark_matter_density","stellar_density","total_density",
              "temperature","hse","gravitational_potential","gravitational_field"]

    for ax,rax,f in zip(axes[::2,:].ravel(),axes[1::2,:].ravel(),fields):
        # - plotting - #
        if f != "hse":
            ax.set_ylabel(f"{f} / {models[0][f].units}")
            rax.set_ylabel(f"res. [dex]")

            for i,model in enumerate(models):

                ax.loglog(model["radius"].d,model[f].d,color=colors[i],ls=ls[i],alpha=0.85)
                rax.loglog(model["radius"].d,(model[f].d - models[1][f].d)/models[1][f].d,color=colors[i],ls=ls[i],alpha=0.85)

                if np.any(model[f].d < 0):
                    ax.set_yscale("symlog")

                rax.set_yscale("symlog")
                rax.set_yticks([-10,-1,0,1,10])
                rax.set_ylim([-10,10])

        else:
            ax.set_ylabel(f"{f}")
            rax.set_ylabel(f"res. [dex]")
            ax.set_ylim([-1,1])
            for i, model in enumerate(models):
                ax.loglog(model["radius"].d, model.check_hse(), color=colors[i], ls=ls[i], alpha=0.85)

                ax.set_yscale("symlog")
    plt.savefig(f"{answer_dir}/rebuild_comp.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Hydrostatic Equilibrium Tests ====================================================================================== #
# -----------------------------                                                                                        #
# These tests are designed to make sure that HSE is consistent between gravity types                                   #
# -------------------------------------------------------------------------------------------------------------------- #
def test_asym_dens_temp(answer_store, answer_dir, asymptotic_models_dens_temp):
    """checks for consistency between"""
    sames = ["density", "entropy", "gravitational_field", "temperature", "total_mass"]
    dat = plot_models(asymptotic_models_dens_temp, np.array([["density", "entropy", "gravitational_field"],
                                              ["temperature", "total_mass", "gravitational_potential"]])
                      , f"{answer_dir}/asym_dens_temp.png", diff=True)

    for k, v in dat.items():
        if k in sames:
            assert_allclose(v[0], v[1], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(asymptotic_models_dens_temp.keys())[0]} and {list(asymptotic_models_dens_temp.keys())[1]} for field {k}")
            assert_allclose(v[0], v[2], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(asymptotic_models_dens_temp.keys())[0]} and {list(asymptotic_models_dens_temp.keys())[2]} for field {k}")


def test_dens_temp(answer_store, answer_dir, standard_models_dens_temp):
    """checks for consistency between"""
    sames = ["density", "entropy", "gravitational_field", "temperature"]
    dat = plot_models(standard_models_dens_temp, np.array([["density", "entropy", "gravitational_field","dark_matter_mass"],
                                                           ["temperature", "total_mass", "gravitational_potential","total_density"]])
                      , f"{answer_dir}/dens_temp.png", diff=True)

    for k, v in dat.items():
        if k in sames:
            assert_allclose(v[0], v[1], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(standard_models_dens_temp.keys())[0]} and {list(standard_models_dens_temp.keys())[1]} for field {k}")
            assert_allclose(v[0], v[2], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(standard_models_dens_temp.keys())[0]} and {list(standard_models_dens_temp.keys())[2]} for field {k}")
def test_asym_dens_tdens(answer_store, answer_dir, asymptotic_models_dens_tdens):
    """checks for consistency between"""
    sames = ["density","total_density","total_mass","dark_matter_density","dark_matter_mass","gravitational_field","temperature"]
    dat = plot_models(asymptotic_models_dens_tdens, np.array([["density", "entropy", "gravitational_field","dark_matter_density"],
                                                             ["temperature", "total_mass", "gravitational_potential","dark_matter_mass"]])
                      , f"{answer_dir}/asym_dens_tdens.png", diff=True)

    for k, v in dat.items():
        if k in sames:
            assert_allclose(v[0], v[1], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(asymptotic_models_dens_tdens.keys())[0]} and {list(asymptotic_models_dens_tdens.keys())[1]} for field {k}")
            assert_allclose(v[0], v[2], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(asymptotic_models_dens_tdens.keys())[0]} and {list(asymptotic_models_dens_tdens.keys())[2]} for field {k}")


def test_dens_tdens(answer_store, answer_dir, standard_models_dens_tdens):
    """checks for consistency between"""
    sames = ["density","total_density","total_mass","dark_matter_density","dark_matter_mass"]
    dat = plot_models(standard_models_dens_tdens, np.array([["density", "entropy", "gravitational_field","dark_matter_density"],
                                                             ["temperature", "total_mass", "gravitational_potential","dark_matter_mass"]])
                      , f"{answer_dir}/dens_tdens.png", diff=True)

    for k, v in dat.items():
        if k in sames:
            assert_allclose(v[0], v[1], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(standard_models_dens_tdens.keys())[0]} and {list(standard_models_dens_tdens.keys())[1]} for field {k}")
            assert_allclose(v[0], v[2], rtol=1e-1,
                            err_msg=f"Failed to meet similarity condition for {list(standard_models_dens_tdens.keys())[0]} and {list(standard_models_dens_tdens.keys())[2]} for field {k}")

#  Virialization Tests
# ----------------------------------------------------------------------------------------------------------------- #
def test_eddington(answer_dir, standard_models_dens_tdens,standard_models_dens_temp):
    # - reading the model - #
    import matplotlib.pyplot as plt
    ms = [standard_models_dens_tdens["Newtonian"],standard_models_dens_temp["Newtonian"]]

    for modelN,type in zip(ms,["dens_tdens","dens_temp"]):
        # - generating the virialization - #
        df = modelN.dm_virial
        pden = modelN[f"dark_matter_density"].d
        rho_check, chk = df.check_virial()

        fig = plt.figure(figsize=(6,10))
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        ax1.loglog(modelN["radius"], pden,color="red",ls="-",lw=2,alpha=0.7)
        ax1.loglog(modelN["radius"], rho_check,color="black",ls="-.",lw=2,alpha=0.7)
        ax2.semilogx(modelN["radius"], chk,color="red",ls="-",lw=2,alpha=0.7)


        ax1.set_title(f"Eddington Formula Test for type={type}")
        ax1.set_ylabel(f"Density {str(modelN['dark_matter_density'].units)}")
        ax2.set_xlabel(f"Radius")
        ax2.set_ylabel("Residuals [dex]")
        ax2.set_yscale("symlog")
        ax2.set_ylim([-1e1,1e1])
        ax1.set_ylim([1e1,1e10])
        ax2.hlines(xmin=np.amin(modelN["radius"].d),xmax=np.amax(modelN["radius"].d),y=0,color="black",ls="-.",lw=3,alpha=0.7)


        fig.savefig(f"{answer_dir}/virial_check_{type}.png")

        assert np.mean(chk[np.where(chk != np.inf)]) < 10e-2, f"Failed to meet virialization criterion ({np.mean(chk[np.where(chk != np.inf)])} > 10e-2) for type={type}."


def test_dispersion_diff(answer_dir, standard_models_dens_tdens,standard_models_dens_temp):
    """Tests the standard dispersions"""
    #  Setup
    # ----------------------------------------------------------------------------------------------------------------- #
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,7))
    ax1,ax2 = fig.add_subplot(121),fig.add_subplot(122)

    for ax,models,type in zip([ax1,ax2],[standard_models_dens_temp,standard_models_dens_tdens],["Density / Temperature","Density / Total Density"]):
        for k,mod in models.items():
            mod.virialization_method="lma"
            mod._dm_virial = None

            vir = mod.dm_virial

            ax.loglog(mod["radius"].d,vir.sigma.to("km**2/s**2").d,color=colors[k],ls=lss[k],alpha=0.75,label=k)


        ax.set_ylabel(r"Velocity Dispersion ($\sigma_r^2$) $\left[\mathrm{km^2\;s^{-2}}\right]$")
        ax.set_xlabel(r"Cluster Radius $\left[\mathrm{kpc}\right]$")
        ax.set_title(f"Dispersion Comparison: {type}")
        ax.legend()
    plt.savefig(f"{answer_dir}/dispersion_comparison.png")
def test_dispersion_asym(answer_dir, asymptotic_models_dens_temp,asymptotic_models_dens_tdens):
    """Tests the asymptotic dispersions"""
    #  Setup
    # ----------------------------------------------------------------------------------------------------------------- #
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,7))
    ax1,ax2 = fig.add_subplot(121),fig.add_subplot(122)

    for ax,models,type in zip([ax1,ax2],[asymptotic_models_dens_temp,asymptotic_models_dens_tdens],["Density / Temperature","Density / Total Density"]):
        for k,mod in models.items():
            mod.virialization_method="lma"
            mod._dm_virial = None

            vir = mod.dm_virial

            ax.loglog(mod["radius"].d,vir.sigma.to("km**2/s**2").d,color=colors[k],ls=lss[k],alpha=0.75,label=k)


        ax.set_ylabel(r"Velocity Dispersion ($\sigma_r^2$) $\left[\mathrm{km^2\;s^{-2}}\right]$")
        ax.set_xlabel(r"Cluster Radius $\left[\mathrm{kpc}\right]$")
        ax.set_title(f"Dispersion Comparison: {type}")
        ax.legend()
    plt.savefig(f"{answer_dir}/dispersion_comparison_asym.png")

def test_total_mass(answer_store, answer_dir):
    """Tests if interp_function: 1 gives the correct information."""
    m, d, r = generate_mdr_potential()
    fields = {"total_mass": m, "radius": r, "total_density": d}

    # - testing parity -#
    potential = Potential(fields, gravity="Newtonian", attrs={"interp_function": lambda x: 1})
    fields["gravitational_field"] = -unyt_array(np.gradient(potential.pot, r.d), units="kpc/Myr**2")

    total_mass_N = _compute_total_mass(fields, gravity="Newtonian", attrs={"interp_function": lambda x: 1})
    total_mass_MA = _compute_total_mass(fields, gravity="AQUAL", attrs={"interp_function": lambda x: 1})
    total_mass_MQ = _compute_total_mass(fields, gravity="QUMOND", attrs={"interp_function": lambda x: 1})

    assert_allclose(total_mass_N.d, total_mass_MQ.d, rtol=1e-3)
    assert_allclose(total_mass_N.d, total_mass_MA.d, rtol=1e-3)


def test_model_temperature(answer_store, answer_dir, asymptotic_models_dens_tdens):
    """Tests if interp_function: 1 gives the correct information."""
    modelN, modelMA, modelMQ = asymptotic_models_dens_tdens["Newtonian"], asymptotic_models_dens_tdens["AQUAL"], asymptotic_models_dens_tdens["QUMOND"]

    assert_allclose(modelN["temperature"], modelMA["temperature"], rtol=1e-3)
    assert_allclose(modelN["temperature"], modelMQ["temperature"], rtol=1e-3)


def test_N_lma(answer_dir, answer_store, standard_models_dens_tdens):
    """Tests the correspondence between LMA and eddington for the Newtonian case."""
    from scipy.interpolate import InterpolatedUnivariateSpline

    #  Creating the particle distributions
    # ----------------------------------------------------------------------------------------------------------------- #
    modelN = standard_models_dens_tdens["Newtonian"]

    parts_eddington = modelN.dm_virial.generate_particles(num_particles=200_000)
    modelN.virialization_method = "lma"
    modelN._dm_virial = None
    parts_lma = modelN.dm_virial.generate_particles(num_particles=200_000)

    parts_lma.make_radial_cut(5000)
    parts_eddington.make_radial_cut(5000)
    parts_eddington.add_offsets(r_ctr=[5000, 5000, 5000], v_ctr=[0, 0, 0])
    parts_lma.add_offsets(r_ctr=[5000, 5000, 5000], v_ctr=[0, 0, 0])

    # -- Creating the YT objects so that we can study them -- #
    pe, plma = parts_eddington.to_yt_dataset(10000), parts_lma.to_yt_dataset(10000)

    # -- Generating the distributions -- #
    profiles, labels = [], ["Eddington Formula", "Local Maxwellian Approx."]
    for ds in [pe, plma]:
        ad = ds.all_data()
        p = yt.create_profile(ad, ("dm", "particle_radius"), ("dm", "particle_velocity_magnitude"),
                              weight_field=("dm", "particle_mass"),
                              units={("dm", "particle_velocity_magnitude"): "km/s"})
        profiles.append(p)

    x = modelN["radius"]
    y = np.sqrt(8 * modelN.dm_virial.sigma / np.pi).to("km/s")

    f, axes = plt.subplots(2, 1, sharex=True, height_ratios=[1, 0.25], gridspec_kw={"hspace": 0})

    # -- plotting for axis 1 -- #
    for p, l, c in zip(profiles, labels, ["forestgreen", "magenta"]):
        m, ca, b = axes[0].errorbar(p.x.to("kpc"), p["dm", "particle_velocity_magnitude"].to("km/s"),
                                    yerr=p.standard_deviation["dm", "particle_velocity_magnitude"], ls=":", marker="+",
                                    capsize=2, color=c, label=l)
        [ci.set_alpha(0.25) for ci in ca];
        [bi.set_alpha(0.25) for bi in b]
        avg_interp = InterpolatedUnivariateSpline(x.d, y.d)
        m, ca, b = axes[1].errorbar(p.x.to("kpc"), (
                    p["dm", "particle_velocity_magnitude"].to("km/s").d - avg_interp(p.x.to("kpc").d)) / avg_interp(
            p.x.to("kpc").d), yerr=p.standard_deviation["dm", "particle_velocity_magnitude"].d / avg_interp(
            p.x.to("kpc").d), ls="", marker="+", capsize=2, color=c, label=l)
        [ci.set_alpha(0.25) for ci in ca];
        [bi.set_alpha(0.25) for bi in b]

    axes[0].loglog(x, y, label=r"$\left<|v|\right>$ (Jean's Equation)")
    axes[1].hlines(xmin=0.1, xmax=1e4, y=0)

    # -- Plot customization -- #
    axes[0].legend()
    axes[1].set_yscale("symlog")
    axes[0].set_ylabel(r"Average Radial Velocity $\left[\mathrm{\frac{km}{s}}\right]$")
    axes[1].set_xlabel(r"Radius $\left[\mathrm{kpc}\right]$")
    axes[1].set_ylabel(r"Residuals $\left[\mathrm{dex}\right]$")
    axes[0].grid()
    axes[1].grid()
    axes[1].set_yticks([-1e1, 0, 1e1])

    # -- assertions -- #
    avg_diff = np.abs(np.mean(
        (p["dm", "particle_velocity_magnitude"].to("km/s").d - avg_interp(p.x.to("kpc").d)) / avg_interp(
            p.x.to("kpc").d)))

    axes[1].text(1e-1, 2, f"Average Deviation = {np.format_float_scientific(avg_diff, precision=3)}",
                 bbox={"ec": "k", "fc": "white", "alpha": 0.7})
    plt.savefig(f"{answer_dir}/lma-edd-validation.png")

    assert avg_diff < 1e-1
