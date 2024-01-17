#----------------------------------------------------------------------------------#
# Cython wrapper for grid based operations
#
# Written by: Eliza Diggins
#----------------------------------------------------------------------------------#
# - structures.pyx provides a cython wrapper for core methods used to write
#   ClusterModel objects to HDF5 grid datasets.
#
#
# FUNCTIONS
#
# interpolate_from_tables(_x,_y)
#   Construct the spline interpolation parameters from the data provied.
#
#
#
#
#

import numpy as np

cimport cython
from cpython cimport array

import array

cimport numpy as np

from scipy.interpolate import dfitpack
from tqdm.auto import tqdm


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

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t


cdef void generate_grid_radii(
        double[3][2] bbox,
        double[3] domain_dimensions,
        double* grid
    ):
    """
    Computes the grid radii for the chunk from the bounding box and the domain dimensions.

    Parameters
    ----------
    bbox: 3x2 array containing the bounding coordinates.
    grid: the blank reference chunk.
    domain_dimensions: the size of the chunk.
    """
    cdef int i=0 ,j=0 ,k=0
    cdef double drx,dry,dyz
    cdef double px=bbox[0][0],py=bbox[1][0],pz=bbox[2][0]

    # compute the grid spacing.

    drx = (bbox[0][1]-bbox[0][0])/domain_dimensions[0]
    dry = (bbox[1][1]-bbox[1][0])/domain_dimensions[1]
    drz = (bbox[2][1]-bbox[2][0])/domain_dimensions[2]


    # compute the grid
    for i in range(domain_dimensions[0]):
        px += drx
        for j in range(domain_dimensions[1]):
            py += dry
            for k in range(domain_dimensions[2]):
                pz += drz
                grid[i][j][k] = sqrt(px*px + py*py + pz*pz)


#=================================================================================#
# Interpolation Routines
#=================================================================================#

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def interpolate_from_tables(
    np.ndarray[DTYPE_t, ndim=1] _x,
    np.ndarray[DTYPE_t, ndim=1] _y
    ):
    """
    Constructs the spline interpolation over ``_x`` and ``_y``.

    Parameters
    ----------
    _x: np.ndarray
        The x values on which to interpolate.
    _y: np.ndarray
        The y values on which to interpolate.

    Returns
    -------

    """
    cdef np.ndarray[DTYPE_t, ndim = 1] t,
    cdef np.ndarray[DTYPE_t, ndim = 1] c,
    cdef int k, n

    _,_,_,_,_,k,_,n,t,c,_,_,_,_ = dfitpack.fpcurf0(_x,_y,3,w=None,xb=_x[0],xe=_x[_x.size-1],s=0.0)
    return t[:n],c[:n],k

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def eval_tck(
    np.ndarray[DTYPE_t, ndim=1] x,
    np.ndarray[DTYPE_t, ndim=1] t,
    np.ndarray[DTYPE_t, ndim=1] c,
    int k
    ):
    return dfitpack.splev(t,c,k,x,0)[0]
