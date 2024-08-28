"""Utilities module for plotting and field label management.

This module provides utilities for plotting, including enforcing a consistent matplotlib style
and managing field labels for various astrophysical quantities.

"""

from functools import wraps

import matplotlib.pyplot as plt

from cluster_generator.utilities.config import cgparams

# Mapping of field names to their plot labels
field_label_map = {
    "density": r"$\rho_g$ (g cm$^{-3}$)",
    "temperature": r"kT (keV)",
    "pressure": r"P (erg cm$^{-3}$)",
    "entropy": r"S (keV cm$^{2}$)",
    "dark_matter_density": r"$\rho_{\rm DM}$ (g cm$^{-3}$)",
    "electron_number_density": r"n$_e$ (cm$^{-3}$)",
    "stellar_mass": r"M$_*$ (M$_\odot$)",
    "stellar_density": r"$\rho_*$ (g cm$^{-3}$)",
    "dark_matter_mass": r"$M_{\rm DM}$ (M$_\odot$)",
    "gas_mass": r"M$_g$ (M$_\odot$)",
    "total_mass": r"M$_{\rm tot}$ (M$_\odot$)",
    "gas_fraction": r"f$_{\rm gas}$",
    "magnetic_field_strength": r"B (G)",
    "gravitational_potential": r"$\Phi$ (kpc$^2$ Myr$^{-2}$)",
    "gravitational_field": r"g (kpc Myr$^{-2}$)",
}
"""dict: A mapping from field names to their corresponding plot labels, formatted for use in matplotlib."""


def enforce_style(func):
    """
    Decorator to enforce a consistent matplotlib style defined in the configuration.

    This decorator temporarily changes the matplotlib rcParams to the default plotting settings
    specified in the configuration (`cgparams.plotting.defaults`). It restores the original
    rcParams after the function is executed.

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The decorated function with enforced matplotlib style.

    Notes
    -----
    This decorator is useful for ensuring that all plots adhere to a consistent style,
    regardless of any changes made to the matplotlib rcParams in the function being decorated.

    Example
    -------
    >>> @enforce_style
    ... def plot_example():
    ...     import matplotlib.pyplot as plt
    ...     plt.plot([0, 1], [0, 1])
    ...     plt.show()

    This will enforce the matplotlib style settings from the configuration before plotting.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Copy the current matplotlib rcParams
        original_rc_params = plt.rcParams.copy()

        # Update rcParams with defaults from the configuration
        for key, value in cgparams.plotting.defaults.items():
            plt.rcParams[key] = value

        try:
            # Execute the function
            result = func(*args, **kwargs)
        finally:
            # Restore the original rcParams after function execution
            plt.rcParams.update(original_rc_params)

        return result

    return wrapper
