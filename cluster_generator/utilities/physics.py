"""Utilities module for physics routines and constants.

This module provides commonly used physical constants and routines for astrophysical calculations.
All constants are represented as `unyt_quantity` objects for seamless unit conversion and manipulation.

"""


from unyt import Unit
from unyt import physical_constants as pc
from unyt import unyt_quantity

from cluster_generator.utilities.config import cgparams

# Constants and conversions
mp: unyt_quantity = pc.mp.to("Msun")
""" :py:class:`unyt.unyt_quantity`: Proton mass in solar masses."""

G: unyt_quantity = pc.G.to("kpc**3/Msun/Myr**2")
""" :py:class:`unyt.unyt_quantity`: Newton's gravitational constant in galactic units."""

kboltz: unyt_quantity = pc.kboltz.to("Msun*kpc**2/Myr**2/K")
""" :py:class:`unyt.unyt_quantity`: Boltzmann's constant in galactic units."""

kpc_to_cm: float = (1.0 * Unit("kpc")).to_value("cm")
"""float: The conversion factor of 1 kpc to centimeters."""

# Hydrogen abundance and mean molecular weights
X_H: float = cgparams.config.physics.hydrogen_abundance
""" float: The cosmological hydrogen abundance.

The adopted value for :math:`X_H` may be changed in the ``cluster_generator`` configuration. Default is 0.76.
"""

mu: float = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
r""" float: The mean molecular mass given the cosmological hydrogen abundance :math:`X_H` and ignoring metals.

.. math::

    \mu = \frac{1}{2X_H + 0.75(1 - X_H)}
"""

mue: float = 1.0 / (X_H + 0.5 * (1.0 - X_H))
r""" float: The mean molecular mass per free electron in a fully ionized primordial plasma.

.. math::

    \mu_e = \frac{1}{X_H + 0.5(1 - X_H)}
"""
