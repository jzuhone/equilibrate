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
cimport numpy as np
cimport cython
from scipy.interpolate import _fitpack
from yt.funcs import get_pbar

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double numer, double denom) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def generate_velocities(np.ndarray[DTYPE_t, ndim=1] psi,
                        np.ndarray[DTYPE_t, ndim=1] vesc,
                        np.ndarray[DTYPE_t, ndim=1] fv2esc,
                        np.ndarray[DTYPE_t, ndim=1] t,
                        np.ndarray[DTYPE_t, ndim=1] c,
                        int k):
    cdef DTYPE_t v2, f, fac
    cdef np.uint8_t not_done
    cdef unsigned int i
    cdef int num_particles, der, ext, ier
    cdef long int seedval
    cdef np.ndarray[np.float64_t, ndim=1] velocity, e
    fac = 1.0/(sqrt(8.)*np.pi**2)
    e = np.zeros(1)
    seedval = -100
    srand48(seedval)
    der = 1
    ext = 0
    num_particles = psi.shape[0]
    velocity = np.zeros(num_particles, dtype='float64')
    pbar = get_pbar("Generating particle velocities", num_particles)
    for i in range(num_particles):
        not_done = 1
        while not_done:
            v2 = drand48()*vesc[i]
            v2 *= v2
            e[0] = psi[i]-0.5*v2
            f = fac*_fitpack._spl_(e, der, t, c, k, ext)[0]
            not_done = f*v2 < drand48()*fv2esc[i]
        velocity[i] = sqrt(v2)
        pbar.update()
    pbar.finish()
    return velocity