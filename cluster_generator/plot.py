"""
These are specialized visualization methods for use in cluster_generator.
"""
from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np
from cluster_generator.utils import mylog, cgparams


# -------------------------------------------------------------------------------------------------------------------- #
# Decorators ========================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
def _enforce_style(func):
    """enforces the mpl style."""

    def wrapper(*args, **kwargs):
        _rcp_copy = plt.rcParams.copy()

        for _k, _v in cgparams["plotting"]["defaults"].items():
            plt.rcParams[_k] = _v

        func(*args, **kwargs)

        plt.rcParams = _rcp_copy
        del _rcp_copy

    return wrapper


# -------------------------------------------------------------------------------------------------------------------- #
# Core Plotting Functions ============================================================================================ #
# -------------------------------------------------------------------------------------------------------------------- #
@_enforce_style
def plot_orbit(orbit, fig=None, title=None, view=None, boxsize=5000, **kwargs):
    """
    Plots the data from ``orbit`` in 3-D and provides plots of the position, velocity, and acceleration.

    Parameters
    ----------
    orbit: SimpleNamespace
        The :py:meth:`ics.ClusterICs.compute_orbits` output to plot.
    fig: Figure, optional
        (``default = None``) The MPL figure to draw the plots in. If none, one is generated.
    title: str, optional
        The title of the plot.
    view: dict, optional
        The dictionary with keys ``az,elev,roll`` indicating the camera position.
    boxsize: int, optional
        The boxsize to enforce. The default is 5000 kpc.
    kwargs: dict, optional
        Additional kwargs to pass through the plotting function.

    Returns
    -------
    Figure
        The output figure.

    """
    #  Setup
    # ---------------------------------------------------------------------------------------------------------------- #
    n_halos = orbit.x.shape[0]
    mylog.info(f"Plotting orbit data for {n_halos} halos.")
    #  Figure / Axes Configuration
    # ---------------------------------------------------------------------------------------------------------------- #
    if fig is None:
        fig = plt.figure(figsize=(10, 7))
    else:
        pass

    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], left=0.00, right=0.995, bottom=0.1, top=0.94, wspace=0.3, hspace=0)
    axes = {
        "main": fig.add_subplot(gs[:, 0], projection="3d"),
        "x"   : fig.add_subplot(gs[0, 1]),
        "dx"  : fig.add_subplot(gs[1, 1]),
        "ddx" : fig.add_subplot(gs[2, 1])
    }

    axis = fig.add_subplot(gs[:, :], frameon=False)
    axis.tick_params("both", bottom=False, top=False, left=False, right=False, labelleft=False, labelright=False,
                     labeltop=False, labelbottom=False)
    if title is None:
        axis.set_title("Orbit Solutions")
    else:
        axis.set_title(title)
    # Configuring kwargs
    # ---------------------------------------------------------------------------------------------------------------- #
    if "markers" not in kwargs:
        markers = ["+", "s", "o"]
    else:
        markers = kwargs["markers"]
        del kwargs["markers"]

    for _k, _v in kwargs.items():
        if hasattr(_v,"__iter__"):
            kwargs[_k] = list(islice(cycle(_v), n_halos))
        else:
            kwargs[_k] = list(islice(cycle([_v]), n_halos))
    #  plotting
    # ---------------------------------------------------------------------------------------------------------------- #
    for h in range(n_halos):
        axes["main"].plot(orbit.x[h, :], orbit.y[h, :], orbit.z[h, :], **{k: v[h] for k, v in kwargs.items()})

        for j in ["", "d", "dd"]:
            for k, i in enumerate(["x", "y", "z"]):
                axes[f"{j}x"].semilogy(orbit.t, getattr(orbit, f"{j}{i}")[h, :], **{k: v[h] for k, v in kwargs.items()})
                axes[f"{j}x"].semilogy(orbit.t[::1000], getattr(orbit, f"{j}{i}")[h, :][::1000], ls="",
                                       marker=markers[k], **{k: v[h] for k, v in kwargs.items() if k != "ls"})

    #  Managing Axes
    # ---------------------------------------------------------------------------------------------------------------- #
    # - scaling - #
    axes["x"].set_yscale("symlog")
    axes["dx"].set_yscale("symlog")
    axes["ddx"].set_yscale("symlog")
    # - grids - #
    axes["x"].grid()
    axes["dx"].grid()
    axes["ddx"].grid()
    # - limits - #
    axes["main"].set_xlim([-boxsize / 2, boxsize / 2])
    axes["main"].set_ylim([-boxsize / 2, boxsize / 2])
    axes["main"].set_zlim([-boxsize / 2, boxsize / 2])
    axes["x"].set_ylim([-boxsize, boxsize])
    axes["ddx"].set_ylim([-1e-2, 1e-2])
    axes["dx"].set_ylim([-10,10])
    # - labels - #
    axes["main"].set_xlabel("x / [kpc]")
    axes["main"].set_ylabel("y / [kpc]")
    axes["main"].set_zlabel("z / [kpc]")
    axes["x"].set_ylabel("Position [kpc]")
    axes["dx"].set_ylabel("Velocity [$\mathrm{kpc\;Myr^{-1}}$]")
    axes["ddx"].set_ylabel("Acceleration [$\mathrm{kpc\;Myr^{-2}}$]")
    axes["ddx"].set_xlabel("Time / [Gyr]")
    # - ticks - #
    axes["ddx"].set_xticklabels([np.round(tick / 1000, decimals=2) for tick in axes["ddx"].get_xticks()])
    axes["x"].tick_params("both", bottom=True, left=True, right=True, top=True, labelbottom=False, labeltop=False,
                          labelright=False)
    axes["dx"].tick_params("both", bottom=True, left=True, right=True, top=True, labelbottom=False, labeltop=False,
                           labelright=False)
    axes["ddx"].tick_params("both", bottom=True, left=True, right=True, top=True, labelbottom=True, labeltop=False,
                            labelright=False)

    if view is not None:
        axes["main"].view_init(**view)
    #  returning
    # ---------------------------------------------------------------------------------------------------------------- #
    return fig
