import numpy as np
from yt.units.yt_array import YTArray
import os
from six import string_types

def parse_value(value, default_units):
    if isinstance(value, YTArray):
        val = YTArray(value.v, value.units).in_units(default_units)
    elif isinstance(value, tuple):
        val = YTArray(value[0], value[1]).in_units(default_units)
    else:
        val = YTArray(value, default_units)
    return val.v

def rot_3d(axis, gx, gy, gz, ang):

    c = np.cos(ang)
    s = np.sin(ang)

    if axis == 1:
        gy, gz = (c*gy + s*gz, -s*gy + c*gz)
    elif axis == 2:
        gx, gz = (c*gx - s*gz, s*gx + c*gz)
    elif axis == 3:
        gx, gy = (c*gx + s*gy, -s*gx + c*gy)

    return gx, gy, gz

class GaussianRandomField(object):
    _units = "dimensionless"
    _name = "vector"

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 alpha=-11./3., g_rms=1.0, ctr1=None, ctr2=None, r1=None,
                 r2=None, g1=None, g2=None, vector_potential=False,
                 divergence_clean=False, prng=np.random):

        self.vector_potential = vector_potential

        le = parse_value(left_edge, "kpc")
        re = parse_value(right_edge, "kpc")

        nx, ny, nz = ddims

        num_halos = 0
        if r1 is not None:
            num_halos += 1
        if r2 is not None:
            num_halos += 2

        if num_halos >= 1:
            if ctr1 is None:
                ctr1 = 0.5*(le+re)
            else:
                ctr1 = parse_value(ctr1, "kpc")
            r1 = parse_value(r1, "kpc")
            g1 = parse_value(g1, self._units)
        if num_halos == 2:
            if ctr2 is None:
                raise RuntimeError("Need to specify 'ctr2' for the second halo!")
            ctr2 = parse_value(ctr2, "kpc")
            r2 = parse_value(r2, "kpc")
            g2 = parse_value(g2, self._units)

        # Derived stuff

        dx, dy, dz = (re-le)/ddims

        l_min = parse_value(l_min, "kpc")
        l_max = parse_value(l_max, "kpc")

        k0 = 2.*np.pi/l_min
        k1 = 2.*np.pi/l_max

        v = np.exp(2.*np.pi*1j*prng.random((3,nx,ny,nz)))

        v[:,0,0,0] = 2.*np.sign((v[:,0,0,0].imag < 0.0).astype("int"))-1.+0j
        v[:,nx//2,ny//2,nz//2] = 2.*np.sign((v[:,nx//2,ny//2,nz//2].imag < 0.0).astype("int"))-1.+0j
        v[:,0,ny//2,nz//2] = 2.*np.sign((v[:,0,ny//2,nz//2].imag < 0.0).astype("int"))-1.+0j
        v[:,nx//2,0,nz//2] = 2.*np.sign((v[:,nx//2,0,nz//2].imag < 0.0).astype("int"))-1.+0j
        v[:,nx//2,ny//2,0] = 2.*np.sign((v[:,nx//2,ny//2,0].imag < np.pi).astype("int"))-1.+0j
        v[:,0,0,nz//2] = 2.*np.sign((v[:,0,0,nz//2].imag < np.pi).astype("int"))-1.+0j
        v[:,0,ny//2,0] = 2.*np.sign((v[:,0,ny//2,0].imag < np.pi).astype("int"))-1.+0j
        v[:,nx//2,0,0] = 2.*np.sign((v[:,nx//2,0,0].imag < np.pi).astype("int"))-1.+0j

        np.multiply(v, np.sqrt(-2.*np.log(prng.random((3,nx,ny,nz)))), v)

        kx,ky,kz = np.mgrid[0:nx,0:ny,0:nz].astype("float64")
        kx[kx > nx//2] = kx[kx > nx//2] - nx 
        ky[ky > ny//2] = ky[ky > ny//2] - ny
        kz[kz > nz//2] = kz[kz > nz//2] - nz
        kx *= 2.*np.pi/(nx*dx)
        ky *= 2.*np.pi/(ny*dy)
        kz *= 2.*np.pi/(nz*dz)
        kk = np.sqrt(kx*kx+ky*ky+kz*kz)
        with np.errstate(invalid='ignore', divide='ignore'):
            sigma = (1.0+(kk/k1)**2)**(0.25*alpha)*np.exp(-(kk/k0)**2)
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

        gx = np.fft.ifftn(sigma*v[0,:,:,:]).real
        gy = np.fft.ifftn(sigma*v[1,:,:,:]).real
        gz = np.fft.ifftn(sigma*v[2,:,:,:]).real

        del sigma, v

        g_avg = np.std(np.sqrt(gx*gx+gy*gy+gz*gz))

        gx /= g_avg
        gy /= g_avg
        gz /= g_avg

        x, y, z = np.mgrid[0:nx,0:ny,0:nz] + 0.5
        x *= dx
        y *= dy
        z *= dz
        x += le[0]
        y += le[1]
        z += le[2]

        if num_halos == 0:
            g_rms = parse_value(g_rms, self._units)
        else:
            if num_halos == 1:
                rr1 = np.sqrt((x-ctr1[0])**2 + (y-ctr1[1])**2 + (z-ctr1[2])**2)
                idxs1 = np.searchsorted(r1, rr1) - 1
                dr1 = (rr1-r1[idxs1])/(r1[idxs1+1]-r1[idxs1])
                g_rms = ((1.-dr1)*g1[idxs1] + dr1*g1[idxs1+1])**2
            if num_halos == 2:
                rr2 = np.sqrt((x-ctr2[0])**2 + (y-ctr2[1])**2 + (z-ctr2[2])**2)
                idxs2 = np.searchsorted(r2, rr2) - 1
                dr2 = (rr2-r2[idxs2])/(r2[idxs2+1]-r2[idxs2])
                g_rms += ((1.-dr2)*g2[idxs2] + dr2*g2[idxs2+1])**2
            g_rms = np.sqrt(g_rms).in_units(self._units).d

        gx *= g_rms
        gy *= g_rms
        gz *= g_rms

        self.data = {"x": YTArray(x[:,0,0], "kpc"),
                     "y": YTArray(y[0,:,0], "kpc"),
                     "z": YTArray(z[0,0,:], "kpc")}

        del x, y, z, g_rms

        if divergence_clean:

            gx = np.fft.fftn(gx)
            gy = np.fft.fftn(gy)
            gz = np.fft.fftn(gz)

            # These k's are different because we are
            # using the finite difference form of the
            # divergence operator.
            kxd = np.sin(kx*dx)/dx
            kyd = np.sin(ky*dy)/dy
            kzd = np.sin(kz*dz)/dz
            kkd = np.sqrt(kxd*kxd+kyd*kyd+kzd*kzd)
            kxd /= kkd
            kyd /= kkd
            kzd /= kkd

            gx, gy, gz = [(1.0-kxd*kxd)*gx-kxd*kyd*gy-kxd*kzd*gz,
                          -kyd*kxd*gx+(1.0-kyd*kyd)*gy-kyd*kzd*gz,
                          -kzd*kxd*gx-kzd*kyd*gy+(1.0-kzd*kzd)*gz]

            gx = np.fft.ifftn(gx).real
            gy = np.fft.ifftn(gy).real
            gz = np.fft.ifftn(gz).real

            del kxd, kyd, kzd, kkd

        if self.vector_potential:

            # Rotate vector potential

            gx = np.fft.fftn(gx)
            gy = np.fft.fftn(gy)
            gz = np.fft.fftn(gz)

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

            gx, gy, gz = rot_3d(3, gx, gy, gz, alpha)
            gx, gy, gz = rot_3d(2, gx, gy, gz, beta)

            with np.errstate(invalid='ignore', divide='ignore'):
                gx, gy = ((0.0+1.0j)*gy/kk, -(0.0+1.0j)*gx/kk)
                gz = np.zeros(gx.shape, dtype="complex")

            gx[np.isinf(gx)] = 0.0
            gx[np.isnan(gx)] = 0.0
            gy[np.isinf(gy)] = 0.0
            gy[np.isnan(gy)] = 0.0

            gx, gy, gz = rot_3d(2, gx, gy, gz,  -beta)
            gx, gy, gz = rot_3d(3, gx, gy, gz, -alpha)

            self.data[self._name+"_x"] = YTArray(np.fft.ifftn(gx).real, "%s*kpc" % self._units)
            self.data[self._name+"_y"] = YTArray(np.fft.ifftn(gy).real, "%s*kpc" % self._units)
            self.data[self._name+"_z"] = YTArray(np.fft.ifftn(gz).real, "%s*kpc" % self._units)

        else:

            self.data[self._name+"_x"] = YTArray(gx, self._units)
            self.data[self._name+"_y"] = YTArray(gy, self._units)
            self.data[self._name+"_z"] = YTArray(gz, self._units)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def units(self):
        if self.vector_potential:
            return "%s*kpc" % self._units
        else:
            return self._units

    def write_to_h5(self, filename, in_cgs=False, overwrite=False):
        import h5py
        if os.path.exists(filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % filename)
        for field in self.data:
            if in_cgs:
                self.data[field].in_cgs().write_hdf5(filename, dataset_name=field)
            else:
                self.data[field].write_hdf5(filename, dataset_name=field)
        f = h5py.File(filename, "r+")
        f.attrs["name"] = self._name
        f.attrs["units"] = self.units
        f.attrs["vector_potential"] = int(self.vector_potential)
        f.flush()
        f.close()

    def map_field_to_particles(self, cluster_particles, ptype="gas", units=None):
        from scipy.interpolate import RegularGridInterpolator
        for i, ax in enumerate("xyz"):
            func = RegularGridInterpolator((self["x"], self["y"], self["z"]),
                                           self[self._name+"_"+ax])
            v = func(cluster_particles["particle_position"][:, 0],
                     cluster_particles["particle_position"][:, 1],
                     cluster_particles["particle_position"][:, 2])
            cluster_particles.set_field(ptype, "particle_%s_%s" % (self._name, ax), v,
                                        units=units)


class RandomMagneticField(GaussianRandomField):
    _units = "gauss"
    _name = "magnetic_field"
    _vector_potential = False

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 alpha=-11./3., B_rms=1.0, ctr1=None, ctr2=None,
                 profile1=None, profile2=None, prng=np.random):
        if profile1 is None:
            r1 = None
            B1 = None
        elif isinstance(profile1, string_types):
            r1 = YTArray.from_hdf5(profile1, dataset_name="radius",
                                   group_name="fields").to('kpc').d
            B1 = YTArray.from_hdf5(profile1, dataset_name="magnetic_field_strength",
                                   group_name="fields")
        else:
            r1, B1 = profile1
        if profile2 is None:
            r2 = None
            B2 = None
        elif isinstance(profile2, string_types):
            r2 = YTArray.from_hdf5(profile2, dataset_name="radius",
                                   group_name="fields").to('kpc').d
            B2 = YTArray.from_hdf5(profile2, dataset_name="magnetic_field_strength",
                                   group_name="fields")
        else:
            r2, B2 = profile2
        super(RandomMagneticField, self).__init__(left_edge, right_edge, ddims,
            l_min, l_max, alpha=alpha, g_rms=B_rms, ctr1=ctr1, ctr2=ctr2, r1=r1,
            r2=r2, g1=B1, g2=B2, divergence_clean=True,
            vector_potential=self._vector_potential, prng=prng)

class RandomMagneticVectorPotential(RandomMagneticField):
    _name = "magnetic_vector_potential"
    _vector_potential = True

class RandomVelocityField(GaussianRandomField):
    _units = "kpc/Myr"
    _name = "velocity"

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 V_rms=1.0, alpha=-11./3., ctr1=None, ctr2=None,
                 profile1=None, profile2=None, divergence_clean=False,
                 prng=np.random):
        if profile1 is None:
            r1 = None
            V1 = None
        elif isinstance(profile1, string_types):
            r1 = YTArray.from_hdf5(profile1, dataset_name="radius",
                                   group_name="fields").d
            V1 = YTArray.from_hdf5(profile1, dataset_name="velocity_dispersion",
                                   group_name="fields")
        else:
            r1, V1 = profile1
        if profile2 is None:
            r2 = None
            V2 = None
        elif isinstance(profile2, string_types):
            r2 = YTArray.from_hdf5(profile2, dataset_name="radius",
                                   group_name="fields").d
            V2 = YTArray.from_hdf5(profile2, dataset_name="velocity_dispersion",
                                   group_name="fields")
        else:
            r2, V2 = profile2
        super(RandomVelocityField, self).__init__(left_edge, right_edge, ddims, 
            l_min, l_max, g_rms=V_rms, alpha=alpha, ctr1=ctr1, ctr2=ctr2, r1=r1,
            r2=r2, g1=V1, g2=V2, divergence_clean=divergence_clean, prng=prng)
