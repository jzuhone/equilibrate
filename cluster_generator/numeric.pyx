#cython: language_level=3, boundscheck=False
"""
Cythonized utilities for equilibrium models
"""
#-------------------------------------------------------------
# Numerical algorithms for profile correction
#
# Eliza C. Diggins
#-------------------------------------------------------------

import numpy as np

cimport cython
cimport numpy as np

import numpy as np


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double numer, double denom) nogil
    double sin(double x) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t

cdef np.ndarray[ITYPE_t, ndim=2] identify_boundaries(np.ndarray[ITYPE_t, ndim=1]hole_array):
    """
    Identifies the boundaries of the ``hole_array``.

    Parameters
    ----------
    hole_array: :py:class:`numpy.ndarray`
        (``TYPE=int32``) the hole array containing ``1`` where there is not a hole and ``0`` where there is a hole.

    Returns
    -------
    h_ids: :py:class:`numpy.ndarray`
        (``Type=int32``) (``shape = (n,2)``). The left and right boundaries of each of the ``n`` identified holes.

    """
    cdef np.ndarray[ITYPE_t, ndim=1] index_array, modified_hole_array, fdifference
    cdef np.ndarray[ITYPE_t, ndim=2] hole_ids
    cdef unsigned long array_size

    # constructing the difference and index arrays
    array_size = hole_array.size + 2

    index_array = np.zeros(array_size, dtype=ITYPE)
    index_array[0], index_array[array_size - 1] = 0, array_size - 3
    index_array[1:array_size - 1] = np.indices([hole_array.size, ])[0]

    modified_hole_array = np.zeros(array_size, dtype=ITYPE)
    modified_hole_array[0], modified_hole_array[array_size - 1] = 1, 1
    modified_hole_array[1:array_size - 1] = hole_array

    # Building the forward difference array
    fdifference = modified_hole_array[1:] - modified_hole_array[:array_size - 1]

    hole_ids = np.zeros((fdifference[np.where(fdifference == -1)].size, 2), dtype=ITYPE)

    hole_ids[:, 0] = index_array[np.where(fdifference == -1)]
    hole_ids[:, 1] = index_array[1:][np.where(fdifference == 1)]

    return hole_ids

cdef np.ndarray[DTYPE_t, ndim=1] _eval_cub_spline(np.ndarray[DTYPE_t, ndim=1]x,
                                                 np.ndarray[DTYPE_t, ndim=1]xs,
                                                 np.ndarray[DTYPE_t, ndim=1]ys,
                                                 np.ndarray[DTYPE_t, ndim=1]dys):
    """
    Evaluates the interpolatory spline values for the HFA algorithm

    Parameters
    ----------
    x: :py:class:`numpy.ndarray`
        x values on which to compute the interpolation.
    xs: :py:class:`numpy.ndarray`
        The x boundaries.
    ys: :py:class:`numpy.ndarray`
        The y values at the boundaries.
    dys: :py:class:`numpy.ndarray`
        The derivatives at the boundaries.

    Returns
    -------
    :py:class:`numpy.ndarray`
        The corresponding interpolation values.

    Notes
    -----
    Form used here is derived from divided difference interpolation to find the Newton form.

    """
    cdef DTYPE_t DELTA, h

    h = xs[1]-xs[0]
    DELTA = (ys[1]-ys[0])/(xs[1]-xs[0])

    return (ys[0]) + \
        (dys[0]*(x-xs[0])) + \
        ((DELTA-dys[0])/h)*(x-xs[0])**2 + \
        ((dys[0]+dys[1]-(2*DELTA))/h**2)*(x-xs[0])*(x-xs[0])*(x-xs[1])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def hia(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    """
    Locates the relevant holes in the function represented by the provided :math:`x` and :math:`y` arrays.

    Parameters
    ----------
    x: :py:class:`numpy.ndarray`
        The domain array
    y: :py:class:`numpy.ndarray`
        The function values at the positions specified in ``x``.

    Returns
    -------

    """
    cdef np.ndarray[DTYPE_t, ndim=1] _x, _y, ymax
    cdef np.ndarray[DTYPE_t, ndim=2] hx, hy
    cdef np.ndarray[ITYPE_t, ndim=2] hole_boundaries
    cdef np.ndarray[ITYPE_t, ndim=1] holes
    cdef unsigned long array_length
    cdef unsigned int j, k

    _x, _y = x[:], y[:]
    array_length = x.size  # Stores array length for manual wrap around indexing.

    # construct the hole array
    holes = np.ones(array_length, dtype=ITYPE)
    ymax = np.maximum.accumulate(_y)

    holes[np.where(~np.isclose(_y, ymax, rtol=1e-4))] = 0

    hole_boundaries = identify_boundaries(holes)

    # Construct the remaining arrays #
    hx, hy = np.zeros((hole_boundaries.shape[0], hole_boundaries.shape[1])), np.zeros(
        (hole_boundaries.shape[0], hole_boundaries.shape[1]))

    for j in range(2):
        for k in range(len(hole_boundaries)):
            hx[k, j], hy[k, j] = _x[hole_boundaries[k, j]], _y[hole_boundaries[k, j]]

    return hole_boundaries, hx, hy

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def hfa(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y,unsigned long max_it):
    """
    Locates the relevant holes in the function represented by the provided :math:`x` and :math:`y` arrays.

    Parameters
    ----------
    x: :py:class:`numpy.ndarray`
        The domain array
    y: :py:class:`numpy.ndarray`
        The function values at the positions specified in ``x``.

    Returns
    -------

    """
    cdef unsigned long arr_size, index, flipped, n
    cdef np.ndarray[DTYPE_t, ndim=1] _x, _y, _dy, _hx, _hy, alpha, beta, xs, ys, ds
    cdef np.ndarray[DTYPE_t, ndim=2] hx, hy
    cdef np.ndarray[ITYPE_t, ndim=2] hi
    cdef np.ndarray[ITYPE_t, ndim=1] _hi, valid_interpolation_indices
    cdef np.ndarray[ITYPE_t, ndim=1] _indx
    arr_size = x.size

    #- enforcing monotonicity direction requirement.
    if y[0] < y[arr_size - 1]:  # quasi-monotone increasing --> leave as is
        flipped = 0
        _x, _y = x[:], y[:]
    elif y[0] > y[arr_size - 1]:
        flipped = 1
        _x, _y = x[:], np.flip(y)
    else:
        raise ValueError("HFA failed because interpolation data contained no discernible trend (y[0] == y[-1]).")

    _indx = np.indices((_x.size,),dtype=ITYPE)[0]

    hi, hx, hy = hia(_x, _y)
    n = 0
    while (len(hi) != 0) & (n < max_it):
        n+=1
        # --> Compute critical iteration values
        _dy = np.gradient(_y, _x)
        _hi, _hx, _hy = hi[0, :], hx[0, :], hy[0, :]

        while (_dy[_hi[0]] < 0) & (_hi[0] > 0):  # --> Forcing a positive derivative at hole edge.
            _hi[0] -= 1

        if _dy[_hi[0]] < 0:
            raise ValueError("HFA failed to locate a positively sloped edge for a hole in the interpolation data.")

        # --> Building the interpolation indicators
        #: alpha = dy/Delta at x, beta = dy/Delta at hx[0] where Delta is the secant slope.
        alpha, beta = _dy[_hi[1]:] / ((_y[_hi[1]:] - _hy[0]) / (_x[_hi[1]:] - _hx[0])), _dy[_hi[0]] / (
                    (_y[_hi[1]:] - _hy[0]) / (_x[_hi[1]:] - _hx[0]))

        valid_interpolation_indices = _indx[_hi[1]:][np.where((beta + alpha < 2) & (beta + alpha > 0))]

        if len(valid_interpolation_indices) == 0:
            # --> Failed to find a valid interpolation edge. We replace with domain edge
            # prefactor on ys avoids numerical issues on forcing monotonicity for flat data.
            # deriv at edge is just relevant secant value at edge.
            xs, ys, ds = np.array([_hx[0], _x[arr_size - 1]]), np.array([_hy[0], 1.01 * np.amax(_y)]), np.array([_dy[_hi[0]],
                                                                                      (ys[1] - ys[0]) / (xs[1] - xs[0])])
            index = arr_size - 1
        else:
            index = valid_interpolation_indices[0]
            xs, ys, ds = np.array([_hx[0], _x[index]]), np.array([_hy[0], _y[index]]), np.array([_dy[_hi[0]], _dy[index]])

        _y[_hi[0]:index+1] = _eval_cub_spline(_x[_hi[0]:index+1],xs,ys,ds)
        hi,hx,hy = hia(_x,_y)

    if n >= max_it:
        raise ValueError(f"HFA completed {n} iterations without success. Returned data may not be monotone.")

    # --> Reverting the direction if necessary
    if flipped == 1:
        _x,_y = _x[:],np.flip(_y)

    return _x,_y
