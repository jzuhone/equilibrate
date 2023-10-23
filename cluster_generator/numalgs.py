"""
Numerical algorithms for use in the backend of the CGP.
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array

from cluster_generator.utils import integrate, mp, mu, mylog


def identify_domain_borders(array, domain=None):
    """
    Identify the edges of the domains specified in the array.

    Parameters
    ----------
    array: :py:class:`numpy.ndarray`
        Array (1D) containing ``1`` indicating truth and ``2`` indicating false from which to obtain the boundaries.
    domain: :py:class:`numpy.ndarray`, optional
        The domain of consideration (x-values) to mark the boundaries instead of using indices.

    Returns
    -------
    list
        List of 2-tuples containing the boundary indices (if ``domain==None``) or the boundary positions if domain is specfied.
    """
    boundaries = (
        np.concatenate([[-1], array[:-1]]) + array + np.concatenate([array[1:], [-1]])
    )
    if domain is None:
        ind = np.indices(array).reshape((array.size,))
        vals = ind[np.where(boundaries == 1)]
    else:
        vals = domain[np.where(boundaries == 1)]

    return vals.reshape(len(vals) // 2, 2)


def _check_non_positive(array, domain=None):
    o = np.zeros(array.size)
    o[np.where(array < 0)] = 1
    return identify_domain_borders(o, domain=domain)


def find_holes(x, y, rtol=1e-3, dy=None):
    """
    Locates holes (points of non-monotonicity) in the profile defined by ``y`` over ``x``.

    Parameters
    ----------
    x: array_like
        The domain array on which to base the hole identification process.
    y: array_like
        The profile to find non-monotonicities in.

        .. warning::

            If the profile is not "nearly monotone increasing" (i.e. ``y[0] >=y[-1]``), then this method will fail.
            If your profile is the wrong way around, simply pass ``y[::-1]`` instead.

    rtol: :obj:`float`, optional
        The hole identification tolerance. Default is ``1e-3``.
    dy: :obj:`callable`, optional
        The derivative function for the array. If it is not provided, a central difference scheme will be used to generate
        the secant slopes.

    Returns
    -------
    n: int
        The number of identified holes.
    h: :py:class:`numpy.ndarray`
        Size ``(3,n,2)`` array containing the following:

        - hx: Array of size ``(n,2)`` with the minimum and maximum ``x`` for each hole respectively.
        - hy: Array of size ``(n,2)`` with the left and right ``y`` value on each side of every hole.
        - hi: Array of size ``(n,2)`` with the left and right indices of the hole.
    """
    _x, _y = x[:], y[:]

    if dy is None:
        secants = np.gradient(_y, _x)
    else:
        secants = dy(_x)

    holes = np.zeros(_x.size)
    ymax = np.maximum.accumulate(_y)
    holes[~np.isclose(_y, ymax, rtol=rtol)] = 1
    holes[secants <= -1e-8] = 1

    # construct boundaries of holes
    _hb = (
        np.concatenate([[holes[0]], holes])[:-1]
        + holes
        + np.concatenate([holes, [holes[-1]]])[1:]
    )
    ind = np.indices(_x.shape).reshape((_x.size,))

    hx, hy, hi = _x[np.where(_hb == 1)], _y[np.where(_hb == 1)], ind[np.where(_hb == 1)]

    if holes[0] == 1:
        hx, hy, hi = (
            np.concatenate([[_x[0]], hx]),
            np.concatenate([[hy[0]], hy]),
            np.concatenate([[0], hi]),
        )
    if holes[-1] == 1:
        hx, hy, hi = (
            np.concatenate([hx, [_x[-1]]]),
            np.concatenate([hy, [hy[-1]]]),
            np.concatenate([hi, [ind[-1]]]),
        )

    return len(hx) // 2, np.array(
        [
            hx.reshape(len(hx) // 2, 2),
            hy.reshape(len(hy) // 2, 2),
            hi.reshape(len(hi) // 2, 2),
        ]
    )


def monotone_interpolation(x, y, buffer=10, rtol=1e-3):
    """
    Monotone interpolation scheme based on piecewise cubic spline methods. Holes are "patched" in such a way that
    the profile is monotone, continuously differentiable (however not 2x continuously differentiable).

    Parameters
    ----------
    x: array-like
        The domain over which interpolation should occur.
    y: array-like
        The relevant profile to force into monotonicity.
    buffer: :obj:`int`, optional
        The forward step buffer. Default is ``10``. Value must be an integer larger than 0.
    rtol: :obj:`float`, optional
        The relative tolerance to enforce on hole identification.

    Returns
    -------
    x: :py:class:`numpy.ndarray`
        The corresponding, newly interpolated, domain.
    y: :py:class:`numpy.ndarray`
        The interpolated solution array.

    Notes
    -----
    Methodology was developed by Eliza Diggins (University of Utah) based on the work of [1]_ and [2]_.

    .. [1] Frisch, F. N. and Carlson, R. E. 1980SJNA...17..238F
    .. [2] Steffen, M. 1990A&A...239..443S

    """
    from scipy.interpolate import CubicHermiteSpline

    if y[-1] > y[0]:
        monotonicity = 1
        _x, _y = x[:], y[:]
    elif y[0] > y[-1]:
        monotonicity = -1
        _x, _y = x[:], y[::-1]
    else:
        mylog.warning(
            "Attempted to find holes in profile with no distinct monotonicity."
        )
        return None

    nholes, holes = find_holes(_x, _y, rtol=rtol)
    derivatives = np.gradient(_y, _x, edge_order=2)

    while nholes > 0:
        # carry out the interpolation over the hole.
        hxx, hyy, hii = holes[:, 0, :]

        # building the interpolant information
        hii[1] = hii[1] + np.min(
            np.concatenate(
                [[buffer, len(_x) - 1 - hii[1]], (holes[2, 1:, 0] - hii[1]).ravel()]
            )
        )
        hii = np.array(hii, dtype="int")
        hyy = [_y[hii[0]], np.amax([_y[hii[1]], _y[hii[0]]])]
        hxx = [_x[hii[0]], _x[hii[1]]]

        if hii[1] == len(_x) - 1:
            print(np.amax(_y))
            _y[hii[0] : hii[1] + 1] = _y[hii[0]]
            print(_y[-10:], hxx, hyy, hii)
            input()
        else:
            xb, yb = hxx[1] - (hyy[1] - hyy[0]) / (2 * derivatives[hii[1]]), (1 / 2) * (
                hyy[0] + hyy[-1]
            )
            s = [(yb - hyy[0]) / (xb - hxx[0]), (hyy[1] - yb) / (hxx[1] - xb)]
            p = (s[0] * (hxx[1] - xb) + (s[1] * (xb - hxx[0]))) / (hxx[1] - hxx[0])
            xs = [hxx[0], xb, _x[hii[-1]]]
            ys = [hyy[0], yb, _y[hii[-1]]]
            dys = [0.0, np.amin([2 * s[0], 2 * s[1], p]), derivatives[hii[1]]]

            cinterp = CubicHermiteSpline(xs, ys, dys)
            _y[hii[0] : hii[1]] = cinterp(_x[hii[0] : hii[1]])

        nholes, holes = find_holes(_x, _y, rtol=rtol)

    if monotonicity == -1:
        _x, _y = _x[:], _y[::-1]

    return _x, _y


def positive_interpolation(x, y, correction_parameter, buffer=10, rtol=1e-3, maxit=10):
    """
    A positive interpolation scheme, which fills "holes" which drop below 0 with 2 piecewise monotone cubics with zero slope at
    the hole center.

    Parameters
    ----------
    x: array-like
        The domain over which interpolation should occur.
    y: array-like
        The relevant profile to force into monotonicity.
    correction_parameter: float
        The correction parameter is a float from 0 to 1 which determines the degree of monotonicity to insist on. If ``correction_parameter == 1``, then
        monotone interpolation will be carried out. If ``correction_parameter == 0``, then the minimum of the function will be at zero over the hole.
    buffer: :obj:`int`, optional
        The step buffer. Default is ``10``. Value must be an integer larger than 0.
    rtol: :obj:`float`, optional
        The relative tolerance to enforce on hole identification.
    maxit: :obj:`int`, optional
        The maximum number of allowed iterations during which an interpolation interval can be found. Default is 10.

    Returns
    -------
    x: :py:class:`numpy.ndarray`
        The corresponding, newly interpolated, domain.
    y: :py:class:`numpy.ndarray`
        The interpolated solution array.

    Notes
    -----
    Methodology was developed by Eliza Diggins (University of Utah) based on the work of [1]_ and [2]_.

    .. [1] Frisch, F. N. and Carlson, R. E. 1980SJNA...17..238F
    .. [2] Steffen, M. 1990A&A...239..443S

    """
    from scipy.interpolate import CubicHermiteSpline

    if correction_parameter == 1:
        return monotone_interpolation(x, y, buffer=buffer, rtol=rtol)

    if y[-1] > y[0]:
        monotonicity = 1
        _x, _y = x[:].copy(), y[:].copy()
    elif y[0] > y[-1]:
        monotonicity = -1
        _x, _y = x[:].copy(), y[::-1].copy()
    else:
        monotonicity = 0
        _x, _y = x[:].copy(), y[:].copy()

    nholes, holes = find_holes(_x, _y, rtol=rtol)
    derivatives = np.gradient(_y, _x, edge_order=2)

    for hid in range(nholes):
        hole = holes[:, hid, :]
        hxx, hyy, hii = hole

        if hii[0] == 0:
            hyy[0] = np.amax([0, hyy[1] / 2])
            derivatives[0] = 0
        n = 0
        check = False
        while (n < maxit) and not check:
            if (monotonicity == 0) or n > 0:
                hii[1] = hii[1] + np.min(
                    np.concatenate(
                        [
                            [buffer, len(_x) - 1 - hii[1]],
                            (holes[2, hid + 1 :, 0] - hii[1]).ravel(),
                        ]
                    )
                )
                hii[0] = hii[0] - np.min(
                    np.concatenate(
                        [[buffer, hii[0]], hii[0] - (holes[2, :hid, 1]).ravel()]
                    )
                )
            else:
                hii[1] = hii[1] + np.min(
                    np.concatenate(
                        [
                            [buffer, len(_x) - 1 - hii[1]],
                            (holes[2, hid + 1 :, 0] - hii[1]).ravel(),
                        ]
                    )
                )
            hii = np.array(hii, dtype="int")
            hyy = [np.amax([_y[hii[0]], 0]), np.amax([_y[hii[1]], 0])]
            hxx = [_x[hii[0]], _x[hii[1]]]

            xb, yb = np.mean(hxx), np.amax([np.mean(hyy), 0]) * correction_parameter
            s = [(yb - hyy[0]) / (xb - hxx[0]), (hyy[1] - yb) / (hxx[1] - xb)]

            if (np.abs(derivatives[hii[0]]) < np.abs(3 * s[0])) and (
                np.abs(derivatives[hii[-1]]) < np.abs(3 * s[1])
            ):
                check = True
            else:
                n += 1

        xs = [hxx[0], xb, hxx[-1]]
        ys = [hyy[0], yb, hyy[-1]]
        dys = [derivatives[hii[0]], 0, derivatives[hii[1]]]

        cinterp = CubicHermiteSpline(xs, ys, dys)
        _y[hii[0] : hii[1]] = cinterp(_x[hii[0] : hii[1]])

    if n >= maxit:
        raise ValueError(
            f"Failed to find a viable interpolation domain within {maxit} iterations."
        )

    if monotonicity == -1:
        _x, _y = _x[:], _y[::-1]

    return _x, _y


def solve_temperature(r, potential_gradient, density):
    """
    Solves the temperature equation from the potential gradient and the gas density.

    Parameters
    ----------
    r: unyt_array
        The radius profile
    potential_gradient: unyt_array
        The potential gradient profile.
    density: unyt_array
        The gas density profile.

    Returns
    -------
    unyt_array:
        The computed temperature profile.

    """
    g = potential_gradient.in_units("kpc/Myr**2").v
    d = density.in_units("Msun/kpc**3").v
    rr = r.in_units("kpc").d
    g_r = InterpolatedUnivariateSpline(rr, g)
    d_r = InterpolatedUnivariateSpline(rr, d)

    dPdr_int = lambda r: d_r(r) * g_r(r)
    P = -integrate(dPdr_int, rr)
    dPdr_int2 = lambda r: d_r(r) * g[-1] * (rr[-1] / r) ** 2
    P -= quad(dPdr_int2, rr[-1], np.inf, limit=100)[0]
    pressure = unyt_array(P, "Msun/kpc/Myr**2")
    temp = pressure * mu * mp / density
    temp.convert_to_units("keV")
    return temp


def _closest_factors(val):
    assert isinstance(val, int), "Value must be integer."

    a, b, i = 1, val, 0

    while a < b:
        i += 1
        if val % i == 0:
            a = i
            b = val // a

    return (a, b)
