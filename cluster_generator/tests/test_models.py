import pytest
import numpy as np
from cluster_generator.model import ClusterModel
from cluster_generator.tests.utils import generate_model_dens_tdens, generate_model_dens_temp, model_answer_testing


# -------------------------------------------------------------------------------------------------------------------- #
# Fixtures =========================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def standard_models_dens_temp(answer_dir, answer_store, gravity):
    import os
    if os.path.exists(f"{answer_dir}/{gravity}_model_dens_temp.h5") and not answer_store:
        return ClusterModel.from_h5_file(f"{answer_dir}/{gravity}_model_dens_temp.h5")
    else:
        m = generate_model_dens_temp(gravity=gravity)
        m.write_model_to_h5(f"{answer_dir}/{gravity}_model_dens_temp.h5", overwrite=True)
        return m


@pytest.fixture
def standard_models_dens_tdens(answer_dir, answer_store, gravity):
    import os
    if os.path.exists(f"{answer_dir}/{gravity}_model_dens_tdens.h5") and not answer_store:
        return ClusterModel.from_h5_file(f"{answer_dir}/{gravity}_model_dens_tdens.h5")
    else:
        m = generate_model_dens_tdens(gravityity=gravity)
        m.write_model_to_h5(f"{answer_dir}/{gravity}_model_dens_tdens.h5", overwrite=True)
    return m


# -------------------------------------------------------------------------------------------------------------------- #
# Construction Tests ================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
def test_model_generation_dens_tdens(answer_store, gravity, answer_dir):
    model = generate_model_dens_tdens(gravity=gravity, attrs={})
    model_answer_testing(model, f"{answer_dir}/{gravity}_model.h5", answer_store, answer_dir)


def test_model_generation_dens_temp(answer_store, gravity, answer_dir):
    model = generate_model_dens_temp(gravity=gravity, attrs={})
    model_answer_testing(model, f"{answer_dir}/{gravity}_model.h5", answer_store, answer_dir)

# -------------------------------------------------------------------------------------------------------------------- #
# Processes Tests ==================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
def test_rebuild(answer_store, answer_dir,gravity):
    """Test that the rebuilding system actually works"""
    import matplotlib.pyplot as plt
    models = [generate_model_dens_temp(gravity=gravity)]
    models.append(models[0].rebuild_physical())

    figure, axes = plt.subplots(6, 4, gridspec_kw={"hspace": 0, "wspace": 0.3}, height_ratios=[1, 0.1, 1, 0.1, 1, 0.1],
                                figsize=(20, 20))
    plt.subplots_adjust(left=0.1, right=.99, bottom=0.1, top=.99)
    colors = ["forestgreen", "black", "magenta"]
    ls = ["-.", ":", "-"]
    fields = ["gas_mass", "dark_matter_mass", "stellar_mass", "total_mass",
              "density", "dark_matter_density", "stellar_density", "total_density",
              "temperature", "hse", "gravitational_potential", "gravitational_field"]

    for ax, rax, f in zip(axes[::2, :].ravel(), axes[1::2, :].ravel(), fields):
        # - plotting - #
        if f != "hse":
            ax.set_ylabel(f"{f} / {models[0][f].units}")
            rax.set_ylabel(f"res. [dex]")

            for i, model in enumerate(models):

                ax.loglog(model["radius"].d, model[f].d, color=colors[i], ls=ls[i], alpha=0.85)
                rax.loglog(model["radius"].d, (model[f].d - models[1][f].d) / models[1][f].d, color=colors[i], ls=ls[i],
                           alpha=0.85)

                if np.any(model[f].d < 0):
                    ax.set_yscale("symlog")

                rax.set_yscale("symlog")
                rax.set_yticks([-10, -1, 0, 1, 10])
                rax.set_ylim([-10, 10])

        else:
            ax.set_ylabel(f"{f}")
            rax.set_ylabel(f"res. [dex]")
            ax.set_ylim([-1, 1])
            for i, model in enumerate(models):
                ax.loglog(model["radius"].d, model.check_hse(), color=colors[i], ls=ls[i], alpha=0.85)

                ax.set_yscale("symlog")
    plt.savefig(f"{answer_dir}/rebuild_comp.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Virialization Tests ================================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
@pytest.mark.skip(reason="See Issue #4 on Github. Currently fails due to non-physical profile behavior.")
def test_eddington(answer_dir, standard_models_dens_tdens, standard_models_dens_temp):
    # - reading the model - #
    import matplotlib.pyplot as plt
    ms = [standard_models_dens_tdens.rebuild_physical(), standard_models_dens_temp.rebuild_physical()]

    for modelN, type in zip(ms, ["dens_tdens", "dens_temp"]):
        # - generating the virialization - #
        df = modelN.dm_virial
        pden = modelN[f"dark_matter_density"].d
        rho_check, chk, res = df.check_virial(rtol=0.1,r_det=2000)

        fig = plt.figure(figsize=(6, 10))
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        ax1.loglog(modelN["radius"], pden, color="red", ls="-", lw=2, alpha=0.7)
        ax1.loglog(modelN["radius"], rho_check, color="black", ls="-.", lw=2, alpha=0.7)
        ax2.semilogx(modelN["radius"], chk, color="red", ls="-", lw=2, alpha=0.7)

        ax1.set_title(f"Eddington Formula Test for type={type}")
        ax1.set_ylabel(f"Density {str(modelN['dark_matter_density'].units)}")
        ax2.set_xlabel(f"Radius")
        ax2.set_ylabel("Residuals [dex]")
        ax2.set_yscale("symlog")
        ax2.set_ylim([-1e1, 1e1])
        ax1.set_ylim([1e1, 1e10])
        ax2.hlines(xmin=np.amin(modelN["radius"].d), xmax=np.amax(modelN["radius"].d), y=0, color="black", ls="-.",
                   lw=3, alpha=0.7)

        fig.savefig(f"{answer_dir}/virial_check_{type}.png")

        assert res

def test_N_lma(answer_dir, answer_store, standard_models_dens_tdens):
    """Tests the correspondence between LMA and eddington for the Newtonian case."""
    from scipy.interpolate import InterpolatedUnivariateSpline
    import yt
    import matplotlib.pyplot as plt
    #  Creating the particle distributions
    # ----------------------------------------------------------------------------------------------------------------- #
    modelN = standard_models_dens_tdens

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
                              units={("dm", "particle_velocity_magnitude"): "km/s"},extrema={("dm","particle_radius"):((10,"kpc"),(5000,"kpc"))})
        profiles.append(p)

    x = modelN["radius"]
    y = np.sqrt(8 * modelN.dm_virial.sigma / np.pi).to("km/s")

    f, axes = plt.subplots(2, 1, sharex=True, height_ratios=[1, 0.25], gridspec_kw={"hspace": 0})

    # -- plotting for axis 1 -- #
    for p, l, c in zip(profiles, labels, ["forestgreen", "magenta"]):
        m, ca, b = axes[0].errorbar(p.x.to("kpc"), p["dm", "particle_velocity_magnitude"].to("km/s"),
                                    yerr=p.standard_deviation["dm", "particle_velocity_magnitude"], ls=":", marker="+",
                                    capsize=2, color=c, label=l)
        [ci.set_alpha(0.25) for ci in ca]
        [bi.set_alpha(0.25) for bi in b]
        avg_interp = InterpolatedUnivariateSpline(x.d, y.d)
        m, ca, b = axes[1].errorbar(p.x.to("kpc"), (
                p["dm", "particle_velocity_magnitude"].to("km/s").d - avg_interp(p.x.to("kpc").d)) / avg_interp(
            p.x.to("kpc").d), yerr=p.standard_deviation["dm", "particle_velocity_magnitude"].d / avg_interp(
            p.x.to("kpc").d), ls="", marker="+", capsize=2, color=c, label=l)
        [ci.set_alpha(0.25) for ci in ca]
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