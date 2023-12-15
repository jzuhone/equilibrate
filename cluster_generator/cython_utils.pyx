#cython: language_level=3, boundscheck=False
"""
Cythonized utilities for equilibrium models
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

cimport cython
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

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def generate_velocities(np.ndarray[DTYPE_t, ndim=1] psi,
                        np.ndarray[DTYPE_t, ndim=1] vesc,
                        np.ndarray[DTYPE_t, ndim=1] fv2esc,
                        np.ndarray[DTYPE_t, ndim=1] t,
                        np.ndarray[DTYPE_t, ndim=1] c,
                        int k):
    cdef DTYPE_t v2, f
    cdef np.uint8_t not_done
    cdef unsigned int i
    cdef int num_particles, der, ext
    cdef long int seedval
    cdef np.ndarray[np.float64_t, ndim=1] velocity, e
    e = np.zeros(1)
    f = 0.0
    seedval = -100
    srand48(seedval)
    der = 0
    ext = 0
    num_particles = psi.shape[0]
    velocity = np.zeros(num_particles, dtype='float64')
    pbar = tqdm(leave=True, total=num_particles,
                desc="Generating particle velocities ")
    for i in range(num_particles):
        not_done = 1
        while not_done:
            v2 = drand48()*vesc[i]
            v2 *= v2
            e[0] = psi[i]-0.5*v2
            f = dfitpack.splev(t,c, k, e, ext)[0][0]
            not_done = f*v2 < drand48()*fv2esc[i]
        velocity[i] = sqrt(v2)
        pbar.update()
    pbar.close()
    return velocity


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def div_clean(np.ndarray[CTYPE_t, ndim=3] gx,
                     np.ndarray[CTYPE_t, ndim=3] gy,
                     np.ndarray[CTYPE_t, ndim=3] gz,
                     np.ndarray[DTYPE_t, ndim=3] kx,
                     np.ndarray[DTYPE_t, ndim=3] ky,
                     np.ndarray[DTYPE_t, ndim=3] kz,
                     np.ndarray[DTYPE_t, ndim=1] deltas):

    cdef int i, j, k
    cdef int nx, ny, nz
    cdef DTYPE_t kxd, kyd, kzd, kkd
    cdef CTYPE_t ggx, ggy, ggz, kg

    nx = gx.shape[0]
    ny = gx.shape[1]
    nz = gx.shape[2]

    # These k's are different because we are
    # using the finite difference form of the
    # divergence operator.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ggx = gx[i,j,k]
                ggy = gy[i,j,k]
                ggz = gz[i,j,k]
                kxd = sin(kx[i,j,k] * deltas[0]) / deltas[0]
                kyd = sin(ky[i,j,k] * deltas[1]) / deltas[1]
                kzd = sin(kz[i,j,k] * deltas[2]) / deltas[2]
                kkd = sqrt(kxd*kxd + kyd*kyd + kzd*kzd)
                if kkd > 0:
                    kxd /= kkd
                    kyd /= kkd
                    kzd /= kkd
                kg = kxd * ggx + kyd * ggy + kzd * ggz
                gx[i,j,k] = ggx - kxd * kg
                gy[i,j,k] = ggy - kyd * kg
                gz[i,j,k] = ggz - kzd * kg
