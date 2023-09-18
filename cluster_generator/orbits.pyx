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
def aqual_orbits(
    np.ndarray[DTYPE_t, ndim=2] x0,
    np.ndarray[DTYPE_t, ndim=2] dx0,
    np.ndarray[DTYPE_t, ndim=1] masses,
    int nhalos,
    DTYPE_t tmax,
    DTYPE_t dtmax,
    DTYPE_t lmax,
    int Nmax,
    DTYPE_t epsilon,
    DTYPE_t rmax,
    ):


    #  Instantiating the necessary arrays / setup procedures
    # ---------------------------------------------------------------------------------------------------------------- #
    cdef long int n # the iteration
    cdef int exit_code
    cdef np.ndarray[DTYPE_t, ndim=3] x
    cdef np.ndarray[DTYPE_t,ndim=1] t
    cdef np.ndarray[DTYPE_t, ndim=3] dx
    cdef np.ndarray[DTYPE_t, ndim=3] ddx
    cdef np.ndarray[DTYPE_t, ndim=2] m_x
    cdef np.ndarray[DTYPE_t, ndim=2] m_dx
    cdef np.ndarray[DTYPE_t, ndim=2] m_ddx
    cdef np.ndarray[DTYPE_t, ndim=1] m_mass
    cdef np.ndarray[DTYPE_t,ndim=2] dt_check
    cdef np.ndarray[DTYPE_t,ndim=2] d
    cdef unsigned long int k
    cdef unsigned long int j
    cdef unsigned long int l
    cdef np.ndarray[unsigned int,ndim=4] mask
    cdef DTYPE_t G
    cdef DTYPE_t dt
    cdef DTYPE_t dt0
    # -- Setup Procedures -- #
    n = 0
    G = 4.49e-12
    a_0 = 0.003868

    dt0 = tmax/float(Nmax)
    dt = tmax/float(Nmax)

    if dt0 > dtmax:
        raise ValueError("Failed to compute integration because dt0 > dtmax.")

    # instantiating the x, dx, ddx variables
    x = np.zeros((3,nhalos,Nmax+1),dtype='float64')
    dx = np.zeros((3,nhalos, Nmax+1), dtype='float64')
    ddx = np.zeros((3,nhalos, Nmax+1), dtype='float64')
    m_x = np.zeros((3,nhalos),dtype='float64')
    m_dx = np.zeros((3,nhalos), dtype='float64')
    m_ddx = np.zeros((3,nhalos), dtype='float64')
    m_mass = np.zeros(nhalos,dtype="float64")
    dt_check = np.zeros((3,nhalos),dtype="float64")
    t = np.zeros(Nmax+1,dtype="float64")
    d = np.zeros((nhalos,nhalos),dtype="float64")

    x[:,:,0] = x0
    dx[:,:,0] = dx0

    # building the mask
    mask = np.zeros((nhalos,3,nhalos,Nmax+1),dtype="uintc")

    for l in range(nhalos):
        for k in range(nhalos):
            if k == l:
                mask[k,:,l,:] = 1
            else:
                mask[k,:,l,:] = 0

    #  Main loop run
    # ---------------------------------------------------------------------------------------------------------------- #
    pbar = tqdm(leave=True, total=Nmax,
                desc="Performing orbital computations")
    while n<Nmax and t[n]<tmax:
        # -- computing forces -- #
        for k in range(nhalos):
            # applying the mask
            m_x,m_dx= np.ma.array(x[:,:,n],mask=mask[k,:,:,n]),np.ma.array(dx[:,:,n],mask=mask[k,:,:,n])
            m_mass = np.ma.array(masses,mask=mask[k,0,:,n])

            # computing the force #
            ddx[:,k,n+1] = G*np.sum((-m_mass[:]*(x[:,k,n,np.newaxis]-m_x[:,:]))/(np.sqrt(np.sum((x[:,k,n,np.newaxis]-m_x[:,:])**2))**(3)),axis=1)

            if np.linalg.norm(ddx[:,k,n]) < a_0:
                ddx[:,k,n+1] = ddx[:,k,n+1]*np.sqrt(a_0/np.linalg.norm(ddx[:,k,n+1]))

            d[k,:] = np.sqrt(np.sum((x[:,:,n] - x[:,k,n,np.newaxis])**2,axis=0))/rmax

        # -- Updating positions -- #

        # - Checking dt - #
        dt_check = (dx[:,:,n]/(2*ddx[:,:,n+1]))*(1+np.sqrt(1+4*(lmax * ddx[:,:,n+1]/dx[:,:,n]**2)))
        dt_check[np.where((np.isnan(dt_check)) | (dt_check < 0))] = dt0

        if np.any(dt_check < dt0):
            dt = np.amin(dt_check)
        else:
            dt = dt0

        # - checking distances - #
        np.fill_diagonal(d,2)
        if np.any(d<1):
            return t[:n], x[:,:,:n], dx[:,:,:n], ddx[:,:,:n], 1

        # - Updating the positions - #
        dx[:,:,n+1] = dx[:,:,n] +  ddx[:,:,n+1]*dt
        x[:,:,n+1] = x[:,:,n] + dx[:,:,n+1]*dt


        #-- Updating the loop --#
        t[n + 1] = t[n] + dt
        n += 1
        pbar.update()
    pbar.close()

    return t,x,dx,ddx,0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def newtonian_orbits(
    np.ndarray[DTYPE_t, ndim=2] x0,
    np.ndarray[DTYPE_t, ndim=2] dx0,
    np.ndarray[DTYPE_t, ndim=1] masses,
    int nhalos,
    DTYPE_t tmax,
    DTYPE_t dtmax,
    DTYPE_t lmax,
    int Nmax,
    DTYPE_t epsilon,
    DTYPE_t rmax,
    ):


    #  Instantiating the necessary arrays / setup procedures
    # ---------------------------------------------------------------------------------------------------------------- #
    cdef long int n # the iteration
    cdef int exit_code
    cdef np.ndarray[DTYPE_t, ndim=3] x
    cdef np.ndarray[DTYPE_t,ndim=1] t
    cdef np.ndarray[DTYPE_t, ndim=3] dx
    cdef np.ndarray[DTYPE_t, ndim=3] ddx
    cdef np.ndarray[DTYPE_t, ndim=2] m_x
    cdef np.ndarray[DTYPE_t, ndim=2] m_dx
    cdef np.ndarray[DTYPE_t, ndim=2] m_ddx
    cdef np.ndarray[DTYPE_t, ndim=1] m_mass
    cdef np.ndarray[DTYPE_t,ndim=2] dt_check
    cdef np.ndarray[DTYPE_t,ndim=2] d
    cdef unsigned long int k
    cdef unsigned long int j
    cdef unsigned long int l
    cdef np.ndarray[unsigned int,ndim=4] mask
    cdef DTYPE_t G
    cdef DTYPE_t dt
    cdef DTYPE_t dt0
    # -- Setup Procedures -- #
    n = 0
    G = 4.49e-12

    dt0 = tmax/float(Nmax)
    dt = tmax/float(Nmax)

    if dt0 > dtmax:
        raise ValueError("Failed to compute integration because dt0 > dtmax.")

    # instantiating the x, dx, ddx variables
    x = np.zeros((3,nhalos,Nmax+1),dtype='float64')
    dx = np.zeros((3,nhalos, Nmax+1), dtype='float64')
    ddx = np.zeros((3,nhalos, Nmax+1), dtype='float64')
    m_x = np.zeros((3,nhalos),dtype='float64')
    m_dx = np.zeros((3,nhalos), dtype='float64')
    m_ddx = np.zeros((3,nhalos), dtype='float64')
    m_mass = np.zeros(nhalos,dtype="float64")
    dt_check = np.zeros((3,nhalos),dtype="float64")
    t = np.zeros(Nmax+1,dtype="float64")
    d = np.zeros((nhalos,nhalos),dtype="float64")

    x[:,:,0] = x0
    dx[:,:,0] = dx0

    # building the mask
    mask = np.zeros((nhalos,3,nhalos,Nmax+1),dtype="uintc")

    for l in range(nhalos):
        for k in range(nhalos):
            if k == l:
                mask[k,:,l,:] = 1
            else:
                mask[k,:,l,:] = 0

    #  Main loop run
    # ---------------------------------------------------------------------------------------------------------------- #
    pbar = tqdm(leave=True, total=Nmax,
                desc="Performing orbital computations")
    while n<Nmax and t[n]<tmax:
        # -- computing forces -- #
        for k in range(nhalos):
            # applying the mask
            m_x,m_dx= np.ma.array(x[:,:,n],mask=mask[k,:,:,n]),np.ma.array(dx[:,:,n],mask=mask[k,:,:,n])
            m_mass = np.ma.array(masses,mask=mask[k,0,:,n])

            # computing the force #
            ddx[:,k,n+1] = G*np.sum((-m_mass[:]*(x[:,k,n,np.newaxis]-m_x[:,:]))/(np.sqrt(np.sum((x[:,k,n,np.newaxis]-m_x[:,:])**2))**(3)),axis=1)

            d[k,:] = np.sqrt(np.sum((x[:,:,n] - x[:,k,n,np.newaxis])**2,axis=0))/rmax

        # -- Updating positions -- #

        # - Checking dt - #
        dt_check = (dx[:,:,n]/(2*ddx[:,:,n+1]))*(1+np.sqrt(1+4*(lmax * ddx[:,:,n+1]/dx[:,:,n]**2)))
        dt_check[np.where((np.isnan(dt_check)) | (dt_check < 0))] = dt0

        if np.any(dt_check < dt0):
            dt = np.amin(dt_check)
        else:
            dt = dt0

        # - checking distances - #
        np.fill_diagonal(d,2)
        if np.any(d<1):
            return t[:n], x[:,:,:n], dx[:,:,:n], ddx[:,:,:n], 1

        # - Updating the positions - #
        dx[:,:,n+1] = dx[:,:,n] +  ddx[:,:,n+1]*dt
        x[:,:,n+1] = x[:,:,n] + dx[:,:,n+1]*dt


        #-- Updating the loop --#
        t[n + 1] = t[n] + dt
        n += 1
        pbar.update()
    pbar.close()

    return t,x,dx,ddx,0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def qumond_orbits(
    np.ndarray[DTYPE_t, ndim=2] x0,
    np.ndarray[DTYPE_t, ndim=2] dx0,
    np.ndarray[DTYPE_t, ndim=1] masses,
    int nhalos,
    DTYPE_t tmax,
    DTYPE_t dtmax,
    DTYPE_t lmax,
    int Nmax,
    DTYPE_t epsilon,
    DTYPE_t rmax,
    ):
    # These are the same because we are forced to linearize the problem... #
    return aqual_orbits(x0,dx0,masses,nhalos,tmax,dtmax,lmax,Nmax,epsilon,rmax)
