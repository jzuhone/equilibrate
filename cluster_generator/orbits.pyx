#cython: language_level=3, boundscheck=False
"""
Orbital Estimation for Newtonian and Non-Newtonian gravity theories

Written by Eliza C. Diggins
"""
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
# Gravity Orbit Solvers ============================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #
# Each of these functions contains a modified Euler's method procedure for estimating the trajectory of the systems
# in the user's initial conditions as point particles.
#
# --> Caveats:
#  - Treats systems as point particles
#  - MOND is linearized.
#
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def newtonian_orbits(
    np.ndarray[DTYPE_t, ndim=1] x0,
    np.ndarray[DTYPE_t, ndim=1] dx0,
    np.ndarray[DTYPE_t, ndim=1] masses,
    int nhalos,
    DTYPE_t eps,
    DTYPE_t dt0,
    DTYPE_t dtmax,
    int Nmax,
    ):
    """

    Parameters
    ----------
    x0
    dx0
    nhalos
    eps
    dt0
    dtmax
    Nmax

    Returns
    -------

    """

    #  Instantiating the necessary arrays / setup procedures
    # ---------------------------------------------------------------------------------------------------------------- #
    cdef long int n # the iteration
    cdef DTYPE_t t # The time variable
    cdef int exit_code
    cdef np.ndarray[DTYPE_t, ndim=3] x
    cdef np.ndarray[DTYPE_t, ndim=3] dx
    cdef np.ndarray[DTYPE_t, ndim=3] ddx
    cdef np.ndarray[DTYPE_t, ndim=2] m_x
    cdef np.ndarray[DTYPE_t, ndim=2] m_dx
    cdef np.ndarray[DTYPE_t, ndim=2] m_ddx
    cdef np.ndarray[DTYPE_t, ndim=1] m_mass
    cdef unsigned long int k
    cdef unsigned long int j
    cdef unsigned long int l
    cdef np.ndarray[int,ndim=4] mask
    cdef DTYPE_t G
    # -- Setup Procedures -- #
    n = 0
    t = 0
    G = 4.49e-12

    # instantiating the x, dx, ddx variables
    x = np.zeros((3,nhalos,Nmax),dtype='float64')
    dx = np.zeros((3,nhalos, Nmax), dtype='float64')
    ddx = np.zeros((3,nhalos, Nmax), dtype='float64')
    m_x = np.zeros((3,nhalos),dtype='float64')
    m_dx = np.zeros((3,nhalos), dtype='float64')
    m_ddx = np.zeros((3,nhalos), dtype='float64')
    m_mass = np.zeros(nhalos,dtype="float64")

    x[:,0] = x0
    dx[:,0] = dx0

    # building the mask
    mask = np.zeros((nhalos,3,nhalos,Nmax),dtype="float64")

    for l in range(nhalos):
        for k in range(nhalos):
            if k == l:
                mask[k,:,l,:] = 1
            else:
                mask[k,:,l,:] = 0

    #  Main loop run
    # ---------------------------------------------------------------------------------------------------------------- #
    while n<Nmax:
        # -- computing forces -- #
        for k in range(nhalos):
            # applying the mask
            m_x,m_dx= np.ma.array(x,mask=mask[k,:,:,n]),np.ma.array(dx,mask=mask[k,:,:,n])
            m_mass = np.ma.array(masses,mask=mask[k,0,:,n])


            # computing the force #
            ddx[:,k,n+1] = G*np.sum((x[:,k,n]-m_x[:,:])/((x[:,k,n]-m_x[:,:])**(3/2)))
            dx[:,k,n+1] = dx[:,k,n+1] +  ddx[:,k,n+1]*dt0
            x[:,k,n+1] = x[:,k,n+1] + dx[:,k,n+1]*dt0

    return x,dx,ddx

