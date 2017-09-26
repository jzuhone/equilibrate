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

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double numer, double denom) nogil
    double pow(double x, double y) nogil
    double floor(double x) nogil

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
                        np.ndarray[DTYPE_t, ndim=1] e,
                        np.ndarray[DTYPE_t, ndim=1] f):
    cdef DTYPE_t v2, ee, fe, r
    cdef np.uint8_t not_done
    cdef unsigned int i, p, j, ne
    cdef int num_particles
    cdef long int seedval
    cdef np.ndarray[np.float64_t, ndim=1] velocity
    seedval = -100
    srand48(seedval)
    num_particles = psi.size
    ne = f.size
    dloge = log10(e[ne-1]/e[0])/ne
    velocity = np.zeros(num_particles, dtype='float64')
    for i in range(num_particles):
        not_done = 1
        while not_done:
            v2 = drand48()*vesc[i]
            v2 *= v2
            ee = psi[i]-0.5*v2
            r = log10(ee/e[0])/dloge
            j = <int>floor(r)
            if r < 0:
                fe = f[0]
            elif j == ne-1:
                fe = f[j]
            else:
                fe = f[j]*pow(f[j+1]/f[j], r-j)
            not_done = fe*v2 < drand48()*fv2esc[i]
        velocity[i] = sqrt(v2)
        p = int(fmod(float(i), float(num_particles/10)))
        if p == 0:
            p = int((100.*i)/float(num_particles)+0.5)
            print("Generated %d%% of particle velocities.\r" % p, end="")
    print("Generated 100% of particle velocities.")
    return velocity
