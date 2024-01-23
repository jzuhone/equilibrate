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
import h5py._hl.dataset
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

ITYPE32 = np.uint32
ctypedef np.uint32_t ITYPE32_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t



##=================================================================================#
## Chunk Management
##=================================================================================#
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def construct_chunks(
    np.ndarray[ITYPE32_t, ndim=1] domain_dimensions,
    unsigned long chunksize
):
    """
    Computes the index bounding boxes on each of the chunks.

    Parameters
    ----------
    domain_dimensions: 3x1 array of ints
    chunksize: uint maximum chunk size.
    """
    #// Compute the number of necessary chunks.
    cdef unsigned long i,j,k,_i
    cdef np.ndarray[ITYPE32_t, ndim=1] nchunks = np.zeros(domain_dimensions.size,dtype="uint32")

    for _i in range(domain_dimensions.size):
        if domain_dimensions[_i]%chunksize == 0:
            nchunks[_i] = domain_dimensions[_i]//chunksize
        else:
            nchunks[_i] = domain_dimensions[_i]//chunksize + 1
    cdef np.ndarray[ITYPE32_t,ndim=5] chunks = np.zeros((3,2,nchunks[0],nchunks[1],nchunks[2]),dtype="uint32")
    cdef np.ndarray[ITYPE32_t,ndim=1] p = np.zeros(3,dtype="uint32")


    for i in range(nchunks[0]):
        p[1:] = 0
        for j in range(nchunks[1]):
            p[2:] = 0
            for k in range(nchunks[2]):
                # figure out the i,j,k - th value
                chunks[:,0,i,j,k] = p
                chunks[:,1,i,j,k] = p+chunksize
                p[2] += chunksize
            chunks[2,1,i,j,nchunks[2]-1] = domain_dimensions[2]
            p[1] += chunksize
        chunks[1, 1, i,  nchunks[1] - 1,:] = domain_dimensions[1]
        p[0] += chunksize
    chunks[0, 1, nchunks[0] - 1,:,:] = domain_dimensions[0]

    return chunks.reshape((3,2,nchunks[0]*nchunks[1]*nchunks[2]))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def dump_field_to_hdf5(
        buffer_object,
        np.ndarray[DTYPE_t, ndim=2] bbox,
        np.ndarray[ITYPE32_t, ndim=1] domain_dimensions,
        np.ndarray[ITYPE32_t, ndim=3] chunkmap,
        np.ndarray[DTYPE_t, ndim=1] t,
        np.ndarray[DTYPE_t, ndim=1] c,
        int k,
        unsigned int level,
        str fieldname
):
    """
    Dump the field to hdf5 buffer.
    """

    cdef np.ndarray[DTYPE_t,ndim=2] dp = ((bbox[:, 1] - bbox[:, 0]) / domain_dimensions).reshape((3,1))
    cdef np.ndarray[DTYPE_t,ndim=2] sbbox
    cdef np.ndarray[ITYPE32_t,ndim=1] s
    cdef np.ndarray[DTYPE_t,ndim=3] _x,_y,_z
    cdef np.ndarray[DTYPE_t,ndim=1] _r
    pbar = tqdm(
        desc=f"Chunked Interpolation: {fieldname}",
        leave=False,
        position=level,
        total=chunkmap.shape[2],
    )

    for chunk_id in range(chunkmap.shape[2]):
        sbbox =  bbox[:, 0].reshape(3, 1) + chunkmap[:,:,chunk_id]*dp
        s = chunkmap[:,1,chunk_id] - chunkmap[:,0,chunk_id]
        _x, _y, _z = np.mgrid[
                     sbbox[0, 0]:sbbox[0, 1]:s[0] * 1j,
                     sbbox[1, 0]:sbbox[1, 1]:s[1] * 1j,
                     sbbox[2, 0]:sbbox[2, 1]:s[2] * 1j
                     ]

        _r = np.sqrt(_x ** 2 + _y ** 2 + _z ** 2).ravel()
        buffer_object[chunkmap[0,0,chunk_id]:chunkmap[0,1,chunk_id],
                                 chunkmap[1,0,chunk_id]:chunkmap[1,1,chunk_id],
                                 chunkmap[2,0,chunk_id]:chunkmap[2,1,chunk_id]] += dfitpack.splev(t,c,k,_r,0)[0].reshape(s)

        pbar.update()
    pbar.close()

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def renormalize(
        buffer_object,
        density_object,
        np.ndarray[ITYPE32_t, ndim=3] chunkmap,
        unsigned int level,
        str fieldname
):
    """
    Dump the field to hdf5 buffer.
    """
    cdef np.ndarray[ITYPE32_t,ndim=1] s
    cdef np.ndarray[DTYPE_t,ndim=3] _x,_y,_z
    cdef np.ndarray[DTYPE_t,ndim=1] _r
    pbar = tqdm(
        desc=f"Normalizing: {fieldname}",
        leave=False,
        position=level,
        total=chunkmap.shape[2],
    )

    for chunk_id in range(chunkmap.shape[2]):

        buffer_object[chunkmap[0,0,chunk_id]:chunkmap[0,1,chunk_id],
                                 chunkmap[1,0,chunk_id]:chunkmap[1,1,chunk_id],
                                 chunkmap[2,0,chunk_id]:chunkmap[2,1,chunk_id]] /= density_object[chunkmap[0,0,chunk_id]:chunkmap[0,1,chunk_id],
                                 chunkmap[1,0,chunk_id]:chunkmap[1,1,chunk_id],
                                 chunkmap[2,0,chunk_id]:chunkmap[2,1,chunk_id]]

        pbar.update()
    pbar.close()

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def unnormalize(
        buffer_object,
        density_object,
        np.ndarray[ITYPE32_t, ndim=3] chunkmap,
        unsigned int level,
        str fieldname
):
    """
    Dump the field to hdf5 buffer.
    """
    cdef np.ndarray[ITYPE32_t,ndim=1] s
    cdef np.ndarray[DTYPE_t,ndim=3] _x,_y,_z
    cdef np.ndarray[DTYPE_t,ndim=1] _r
    pbar = tqdm(
        desc=f"Un-normalizing: {fieldname}",
        leave=False,
        position=level,
        total=chunkmap.shape[2],
    )

    for chunk_id in range(chunkmap.shape[2]):

        buffer_object[chunkmap[0,0,chunk_id]:chunkmap[0,1,chunk_id],
                                 chunkmap[1,0,chunk_id]:chunkmap[1,1,chunk_id],
                                 chunkmap[2,0,chunk_id]:chunkmap[2,1,chunk_id]] *= density_object[chunkmap[0,0,chunk_id]:chunkmap[0,1,chunk_id],
                                 chunkmap[1,0,chunk_id]:chunkmap[1,1,chunk_id],
                                 chunkmap[2,0,chunk_id]:chunkmap[2,1,chunk_id]]

        pbar.update()
    pbar.close()
