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
from tqdm.auto import tqdm


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double numer, double denom) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# -------------------------------------------------------------------------------------------------------------------- #
# Eddington Formula + Von Neumann Sampling =========================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def generate_velocities(np.ndarray[DTYPE_t, ndim=1] psi,
                        np.ndarray[DTYPE_t, ndim=1] vesc,
                        np.ndarray[DTYPE_t, ndim=1] fv2esc,
                        np.ndarray[DTYPE_t, ndim=1] t,
                        np.ndarray[DTYPE_t, ndim=1] c,
                        int k):
    # Generates the velocities associated with the particle family #
    #  Allocating variable namespace / types
    # ----------------------------------------------------------------------------------------------------------------- #
    cdef DTYPE_t v2, f
    cdef np.uint8_t not_done
    cdef unsigned int i
    cdef int num_particles, der, ext
    cdef long int seedval
    cdef np.ndarray[np.float64_t, ndim=1] velocity, e

    e = np.zeros(1)  # used to hold the energy value

    # - Starting the PRNG.
    seedval = -100
    srand48(seedval)
    der = 0
    ext = 0
    num_particles = psi.shape[0]
    velocity = np.zeros(num_particles, dtype='float64')

    #  Executing
    # ----------------------------------------------------------------------------------------------------------------- #
    pbar = tqdm(leave=True, total=num_particles,
                desc="Generating particle velocities [Eddington]")
    for i in range(num_particles):
        not_done = 1
        while not_done:
            v2 = drand48() * vesc[i]
            v2 *= v2
            e[0] = psi[i] - 0.5 * v2
            f = _fitpack._spl_(e, der, t, c, k, ext)[0]
            not_done = f * v2 < drand48() * fv2esc[i]
        velocity[i] = sqrt(v2)
        pbar.update()
    pbar.close()
    return velocity
# -------------------------------------------------------------------------------------------------------------------- #
# Local Maxwellian Approximation ===================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def generate_lma_velocities(np.ndarray[DTYPE_t, ndim=1] disp,
                            np.ndarray[DTYPE_t, ndim=1] vesc,
                            np.ndarray[DTYPE_t, ndim=1] t,
                            np.ndarray[DTYPE_t, ndim=1] c,
                            int k,
                            double alpha
                            ):
    """
    Generates the Local Maxwellian Approximation velocities.

    Parameters
    ----------
    disp: np.ndarray
        The dispersion array should contain the LMA specific dispersions for each of the particles. Particles are initialized
        outside of the cython module and then for each position, the dispersion is read from spline and passed here.
    vesc: np.ndarray
        The escape velocity of the particle matching the index. This also should be prepopulated.
    t: np.ndarray
        Distribution spline parameter.
    c: np.ndarray
        Distribution spline parameter.
    k: int
        Distribution spline parameter: the degree of the spline.
    alpha: double
        The cutoff fraction for the velocity.

    Returns
    -------
    velocity: np.ndarray
        The velocity (speed) array for the particles.
    """
    #  Declaring locals and setting base values
    # ----------------------------------------------------------------------------------------------------------------- #
    cdef DTYPE_t v2, f
    cdef np.uint8_t not_done
    cdef unsigned int i
    cdef int num_particles, der, ext
    cdef long int seedval
    cdef np.ndarray[np.float64_t, ndim=1] velocity, e
    e = np.zeros(1)  # used to hold the sample value
    seedval = -100  # The seed for the random number generator.
    srand48(seedval)  # Initializing the random number generator.
    der = 0  # The spline derivative (0 -> no derivatives needed)
    ext = 0  # The out of bounds behavior of the spline.
    num_particles = disp.shape[0]
    velocity = np.zeros(num_particles, dtype='float64')
    #  Computing values
    # ----------------------------------------------------------------------------------------------------------------- #

    pbar = tqdm(leave=True, total=num_particles,
                desc="Generating particle velocities [LMA]")

    for i in range(num_particles):
        # --- Seeking a value for the velocity of the ith particle --- #
        not_done = 1
        while not_done:
            e[0] = drand48()  # randomly sampled value.
            # --- sampling from the inverse spline --- #
            f = _fitpack._spl_(e, der, t, c, k, ext)[0]
            # --- checking --- #
            not_done = sqrt(2.0) * sqrt(disp[i]) * f > alpha * vesc[i]
        velocity[i] = sqrt(2.0) * sqrt(disp[i]) * f
        pbar.update()
    pbar.close()

    return velocity
