import numpy as np
from yt.units.yt_array import YTArray
from yt.funcs import iterable

def parse_value(value, default_units):
    if isinstance(value, YTArray):
        return YTArray(value.v, value.units).in_units(default_units)
    elif iterable(value):
        return YTArray(value[0], value[1]).in_units(default_units)
    else:
        return YTArray(value, default_units)

def rot_3d(axis, x1, y1, z1, ang):

    c = np.cos(ang)
    s = np.sin(ang)

    if axis == 1:
        x2 =  x1
        y2 =  c*y1 + s*z1
        z2 = -s*y1 + c*z1
    elif axis == 2:
        x2 = c*x1 - s*z1
        y2 = y1
        z2 = s*x1 + c*z1
    elif axis == 3:
        x2 =  c*x1 + s*y1
        y2 = -s*x1 + c*y1
        z2 = z1

    return x2, y2, z2

def gaussian_random_field(left_edge, right_edge, ddims, l_min, l_max,
                          avg_A=1.0, ctr1=None, ctr2=None, r1=None,
                          r2=None, A1=None, A2=None):

    le = parse_value(left_edge, "kpc").v
    re = parse_value(right_edge, "kpc").v

    nx, ny, nz = ddims

    num_halos = 0
    if ctr1 is not None:
        ctr1 = parse_value(ctr1, "kpc").v
        num_halos = 1
    if ctr2 is not None:
        num_halos = 2
        ctr2 = parse_value(ctr2, "kpc").v

    # Derived stuff

    dx, dy, dz = (re-le)/ddims

    l_min = parse_value(l_min, "kpc").v
    l_max = parse_value(l_max, "kpc").v

    k0 = 2.*np.pi/l_min
    k1 = 2.*np.pi/l_max

    a = np.sqrt(-2.*np.log(np.random.random((3,nx,ny,nz))))
    phi = 2.*np.pi*np.random.random((3,nx,ny,nz))
    v = np.cos(phi)+1j*np.sin(phi)

    v[:,0,0,0] = 2.*np.sign((phi[:,0,0,0] < np.pi).astype("int"))-1.+0j
    v[:,nx//2,ny//2,nz//2] = 2.*np.sign((phi[:,nx//2,ny//2,nz//2] < np.pi).astype("int"))-1.+0j
    v[:,0,ny//2,nz//2] = 2.*np.sign((phi[:,0,ny//2,nz//2] < np.pi).astype("int"))-1.+0j
    v[:,nx//2,0,nz//2] = 2.*np.sign((phi[:,nx//2,0,nz//2] < np.pi).astype("int"))-1.+0j
    v[:,nx//2,ny//2,0] = 2.*np.sign((phi[:,nx//2,ny//2,0] < np.pi).astype("int"))-1.+0j
    v[:,0,0,nz//2] = 2.*np.sign((phi[:,0,0,nz//2] < np.pi).astype("int"))-1.+0j
    v[:,0,ny//2,0] = 2.*np.sign((phi[:,0,ny//2,0] < np.pi).astype("int"))-1.+0j
    v[:,nx//2,0,0] = 2.*np.sign((phi[:,nx//2,0,0] < np.pi).astype("int"))-1.+0j

    del phi

    v *= a

    del a

    kx,ky,kz = np.mgrid[0:nx,0:ny,0:nz].astype("float64")
    kx[kx > nx//2] = kx[kx > nx//2] - nx 
    ky[ky > ny//2] = ky[ky > ny//2] - ny
    kz[kz > nz//2] = kz[kz > nz//2] - nz
    kx *= 2.*np.pi/(nx*dx)
    ky *= 2.*np.pi/(ny*dy)
    kz *= 2.*np.pi/(nz*dz)
    kk = np.sqrt(kx*kx+ky*ky+kz*kz)
    with np.errstate(invalid='ignore', divide='ignore'):
        sigma = np.exp(-(k1/kk)**2)*(kk**(-11./6.))*np.exp(-(kk/k0)**2)
    sigma[np.isinf(sigma)] = 0.0
    sigma[np.isnan(sigma)] = 0.0

    v[:,nx-1:0:-1,ny-1:0:-1,nz-1:nz//2:-1] = np.conjugate(v[:,1:nx,1:ny,1:nz//2])
    v[:,nx-1:0:-1,ny-1:ny//2:-1,nz//2] = np.conjugate(v[:,1:nx,1:ny//2,nz//2])
    v[:,nx-1:0:-1,ny-1:ny//2:-1,0] = np.conjugate(v[:,1:nx,1:ny//2,0])
    v[:,nx-1:0:-1,0,nz-1:nz//2:-1] = np.conjugate(v[:,1:nx,0,1:nz//2])
    v[:,0,ny-1:0:-1,nz-1:nz//2:-1] = np.conjugate(v[:,0,1:ny,1:nz//2])
    v[:,nx-1:nx//2:-1,ny//2,nz//2] = np.conjugate(v[:,1:nx//2,ny//2,nz//2])
    v[:,nx-1:nx//2:-1,ny//2,0] = np.conjugate(v[:,1:nx//2,ny//2,0])
    v[:,nx-1:nx//2:-1,0,nz//2] = np.conjugate(v[:,1:nx//2,0,nz//2])
    v[:,0,ny-1:ny//2:-1,nz//2] = np.conjugate(v[:,0,1:ny//2,nz//2])
    v[:,nx-1:nx//2:-1,0,0] = np.conjugate(v[:,1:nx//2,0,0])
    v[:,0,ny-1:ny//2:-1,0] = np.conjugate(v[:,0,1:ny//2,0])
    v[:,0,0,nz-1:nz//2:-1] = np.conjugate(v[:,0,0,1:nz//2])

    ax = np.real(np.fft.ifftn(sigma*v[0,:,:,:]))
    ay = np.real(np.fft.ifftn(sigma*v[1,:,:,:]))
    az = np.real(np.fft.ifftn(sigma*v[2,:,:,:]))

    del sigma, v

    Aavg = np.std(np.sqrt(ax*ax+ay*ay+az*az))

    ax /= Aavg
    ay /= Aavg
    az /= Aavg

    x,y,z = np.mgrid[0:nx,0:ny,0:nz] + 0.5
    x *= dx
    y *= dy
    z *= dz
    x += le[0]
    y += le[1]
    z += le[2]

    if num_halos == 0:
        avg_A = parse_value(avg_A, "gauss").v
    else:
        if num_halos == 1:
            rr1 = np.sqrt((x-ctr1[0])**2 + (y-ctr1[1])**2 + (z-ctr1[2])**2)
            idxs1 = np.searchsorted(r1, rr1) - 1
            dr1 = (rr1-r1[idxs1])/(r1[idxs1+1]-r1[idxs1])
            avg_A = ((1.-dr1)*A1[idxs1] + dr1*A1[idxs1+1])**2
        if num_halos == 2:
            rr2 = np.sqrt((x-ctr2[0])**2 + (y-ctr2[1])**2 + (z-ctr2[2])**2)
            idxs2 = np.searchsorted(r2, rr2) - 1
            dr2 = (rr2-r2[idxs2])/(r2[idxs2+1]-r2[idxs2])
            avg_A += ((1.-dr2)*A2[idxs2] + dr2*A2[idxs2+1])**2
            avg_A = np.sqrt(avg_A)

    ax *= avg_A
    ay *= avg_A
    az *= avg_A

    field_units = str(avg_A.units)

    axcoord = YTArray(x[:,0,0], "kpc")
    aycoord = YTArray(y[0,:,0], "kpc")
    azcoord = YTArray(z[0,0,:], "kpc")

    del x,y,z,avg_A

    # Rotate vector potential

    axk = np.fft.fftn(ax)
    ayk = np.fft.fftn(ay)
    azk = np.fft.fftn(az)

    with np.errstate(invalid='ignore', divide='ignore'):
        alpha = np.arccos(kx/np.sqrt(kx*kx+ky*ky))
    alpha[ky < 0.0] -= 2.0*np.pi
    alpha[ky < 0.0] *= -1.
    with np.errstate(invalid='ignore', divide='ignore'):
        beta = np.arccos(kz/kk)
    alpha[np.isinf(alpha)] = 0.0
    alpha[np.isnan(alpha)] = 0.0
    beta[np.isnan(beta)] = 0.0
    beta[np.isinf(beta)] = 0.0

    axk, ayk, azk = rot_3d(3, axk, ayk, azk, alpha)
    axk, ayk, azk = rot_3d(2, axk, ayk, azk, beta)

    with np.errstate(invalid='ignore', divide='ignore'):
        axk = (0.0+1.0j)*ayk/kk
        ayk = -(0.0+1.0j)*axk/kk
        azk = np.zeros(axk.shape, dtype="complex")

    axk[np.isinf(axk)] = 0.0
    axk[np.isnan(axk)] = 0.0
    ayk[np.isinf(ayk)] = 0.0
    ayk[np.isnan(ayk)] = 0.0

    axk, ayk, azk = rot_3d(2, axk, ayk, azk,  -beta)
    axk, ayk, azk = rot_3d(3, axk, ayk, azk, -alpha)

    ax = YTArray(np.real(np.fft.ifftn(axk)), "%s*kpc" % field_units)
    ay = YTArray(np.real(np.fft.ifftn(ayk)), "%s*kpc" % field_units)
    az = YTArray(np.real(np.fft.ifftn(azk)), "%s*kpc" % field_units)

    return axcoord, aycoord, azcoord, ax, ay, az

