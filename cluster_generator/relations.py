"""Physical relationships taken from theory or from literature."""
import numpy as np
import unyt

from cluster_generator.utils import ensure_ytquantity, mue


def f_gas(m500, hubble: float = 0.7) -> float:
    r"""
    The relationship between M500 and f_gas = Mgas/Mtot
    within r500 from Vikhlinin, A., et al. 2009, ApJ,
    692, 1033 (https://ui.adsabs.harvard.edu/abs/2009ApJ...692.1033V/).

    Parameters
    ----------
    m500 : float or :py:class:`unyt.unyt_quantity`
        The :math:`M_{500}` if the cluster.
    hubble : float, optional
        The Hubble parameter in units of 100 km/s/Mpc.
        Default: 0.7

    Returns
    -------
    float
        The ICM gas fraction at :math:`r_{500}`.
    """
    m500 = ensure_ytquantity(m500, "Msun").to_value("Msun")
    m = m500 * 1.0e-15 / hubble

    return ((0.72 / hubble) ** 1.5) * (0.125 + 0.037 * np.log10(m))


def m_bcg(m500) -> unyt.unyt_quantity:
    r"""Determine the BCG mass given the cluster's :math:`M_{500}`.

    Parameters
    ----------
    m500: :py:class:`unyt.unyt_quantity`
        The :math:`M_{500}` value.

    Returns
    -------
    :py:class:`unyt.unyt_quantity`
        The resulting BCG mass.

    Notes
    -----
    Utilizes the relation

    .. math::

        \log M_{\mathrm{BCG}} = \left(0.39 \times \left[\log M_{500} - 14.5\right]\right) + 12.15
    """
    # TODO: What's the source on this?
    m500 = ensure_ytquantity(m500, "Msun").to_value("Msun")
    x = np.log10(m500) - 14.5
    y = 0.39 * x + 12.15
    return unyt.unyt_quantity(10**y, "Msun")


def m_sat(m500) -> unyt.unyt_quantity:
    r"""Determine the sat mass given the cluster's :math:`M_{500}`.

    Parameters
    ----------
    m500: :py:class:`unyt.unyt_quantity`
        The :math:`M_{500}` value.

    Returns
    -------
    :py:class:`unyt.unyt_quantity`
        The resulting sat mass.

    Notes
    -----
    Utilizes the relation

    .. math::

        \log M_{\mathrm{BCG}} = \left(0.87 \times \left[\log M_{500} - 14.5\right]\right) + 12.42
    """
    # TODO: What is sat here? Also, what's the source.
    m500 = ensure_ytquantity(m500, "Msun").to_value("Msun")
    x = np.log10(m500) - 14.5
    y = 0.87 * x + 12.42
    return 10**y


def r_bcg(r200) -> unyt.unyt_quantity:
    r"""Determine the BCG radius given the cluster's :math:`r_{200}`.

    Parameters
    ----------
    r200: :py:class:`unyt.unyt_quantity`
        The :math:`r_{200}` value.

    Returns
    -------
    :py:class:`unyt.unyt_quantity`
        The resulting BCG radius.

    Notes
    -----
    Utilizes the relation

    .. math::

        \log M_{\mathrm{BCG}} = \left(0.95 \times \left[\log r_{200} - 1.0\right]\right) - 0.3
    """
    # TODO: Source
    r200 = ensure_ytquantity(r200, "kpc").to_value("kpc")
    x = np.log10(r200) - 1.0
    y = 0.95 * x - 0.3
    return unyt.unyt_quantity(10**y, "kpc")


def convert_ne_to_density(ne) -> unyt.unyt_quantity:
    r"""Convert the electron number density into the corresponding mass density.

    Parameters
    ----------
    ne: :py:class:`unyt.unyt_quantity`
        The electron number density.

    Returns
    -------
    :py:class:`unyt.unyt_quantity`
        The corresponding gas density.

    Notes
    -----
    This is a simple unit conversion using the pre-determined :math:`\mu_e` value.
    """
    ne = ensure_ytquantity(ne, "cm**-3")
    return ne.to("Msun/kpc**3", "number_density", mu=mue)
