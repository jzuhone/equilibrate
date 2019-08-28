import numpy as np
from yt.units.yt_array import YTArray
from yt import mylog
import os
from six import string_types


def parse_value(value, default_units):
    if isinstance(value, YTArray):
        val = YTArray(value.v, value.units).in_units(default_units)
    elif isinstance(value, tuple):
        val = YTArray(value[0], value[1]).in_units(default_units)
    else:
        val = YTArray(value, default_units)
    return val


def rot_3d(axis, gx, gy, gz, ang):

    c = np.cos(ang)
    s = np.sin(ang)

    if axis == 1:
        gy, gz = c*gy + s*gz, -s*gy + c*gz
    elif axis == 2:
        gx, gz = c*gx - s*gz, s*gx + c*gz
    elif axis == 3:
        gx, gy = c*gx + s*gy, -s*gx + c*gy

    return gx, gy, gz


class ClusterField(object):
    _units = "dimensionless"
    _name = "vector"

    def __init__(self, left_edge, right_edge, ddims, vector_potential=False,
                 divergence_clean=False):

        self.left_edge = parse_value(left_edge, "kpc").v
        self.right_edge = parse_value(right_edge, "kpc").v
        self.ddims = ddims
        self.vector_potential = vector_potential
        self.divergence_clean = divergence_clean
        self.comps = ["{}{}".format(self._name, ax) for ax in "xyz"]
        self.dx, self.dy, self.dz = (self.right_edge-self.left_edge)/self.ddims

    def _compute_coords(self):
        nx, ny, nz = self.ddims
        x, y, z = np.mgrid[0:nx,0:ny,0:nz] + 0.5
        x *= self.dx
        y *= self.dy
        z *= self.dz
        x += self.left_edge[0]
        y += self.left_edge[1]
        z += self.left_edge[2]
        return x, y, z

    def _compute_waves(self):
        nx, ny, nz = self.ddims

        kx, ky, kz = np.mgrid[0:nx,0:ny,0:nz].astype("float64")
        kx[kx > nx//2] = kx[kx > nx//2] - nx
        ky[ky > ny//2] = ky[ky > ny//2] - ny
        kz[kz > nz//2] = kz[kz > nz//2] - nz
        kx *= 2.*np.pi/(nx*self.dx)
        ky *= 2.*np.pi/(ny*self.dy)
        kz *= 2.*np.pi/(nz*self.dz)

        return kx, ky, kz

    def _rot_3d(self, axis, ang):
        c = np.cos(ang)
        s = np.sin(ang)

        if axis == 1:
            self.gy, self.gz = c*self.gy+s*self.gz, -s*self.gy+c*self.gz
        elif axis == 2:
            self.gx, self.gz = c*self.gx-s*self.gz, s*self.gx+c*self.gz
        elif axis == 3:
            self.gx, self.gy = c*self.gx+s*self.gy, -s*self.gx+c*self.gy

    def _divergence_clean(self, kx, ky, kz):

        mylog.info("Perform divergence cleaning.")

        self.gx = np.fft.fftn(self.gx)
        self.gy = np.fft.fftn(self.gy)
        self.gz = np.fft.fftn(self.gz)

        # These k's are different because we are
        # using the finite difference form of the
        # divergence operator.
        kxd = np.sin(kx * self.dx) / self.dx
        kyd = np.sin(ky * self.dy) / self.dy
        kzd = np.sin(kz * self.dz) / self.dz
        kkd = np.sqrt(kxd * kxd + kyd * kyd + kzd * kzd)
        with np.errstate(invalid='ignore', divide='ignore'):
            kxd /= kkd
            kyd /= kkd
            kzd /= kkd
            kxd[np.isnan(kxd)] = 0.0
            kyd[np.isnan(kyd)] = 0.0
            kzd[np.isnan(kzd)] = 0.0

        self.gx, self.gy, self.gz = \
            [(1.0-kxd*kxd)*self.gx-kxd*kyd*self.gy-kxd*kzd*self.gz,
             -kyd*kxd*self.gx+(1.0-kyd*kyd)*self.gy-kyd*kzd*self.gz,
             -kzd*kxd*self.gx-kzd*kyd*self.gy+(1.0-kzd*kzd)*self.gz]

        self.gx = np.fft.ifftn(self.gx).real
        self.gy = np.fft.ifftn(self.gy).real
        self.gz = np.fft.ifftn(self.gz).real

        del kxd, kyd, kzd, kkd

    def _compute_vector_potential(self, kx, ky, kz):

        kk = np.sqrt(kx**2+ky**2+kz**2)

        mylog.info("Compute vector potential.")

        # Rotate vector potential

        self.gx = np.fft.fftn(self.gx)
        self.gy = np.fft.fftn(self.gy)
        self.gz = np.fft.fftn(self.gz)

        with np.errstate(invalid='ignore', divide='ignore'):
            alpha = np.arccos(kx / np.sqrt(kx * kx + ky * ky))
        alpha[ky < 0.0] -= 2.0 * np.pi
        alpha[ky < 0.0] *= -1.
        with np.errstate(invalid='ignore', divide='ignore'):
            beta = np.arccos(kz / kk)
        alpha[np.isinf(alpha)] = 0.0
        alpha[np.isnan(alpha)] = 0.0
        beta[np.isnan(beta)] = 0.0
        beta[np.isinf(beta)] = 0.0

        self._rot_3d(3, alpha)
        self._rot_3d(2, beta)

        with np.errstate(invalid='ignore', divide='ignore'):
            self.gx, self.gy = (0.0+1.0j)*self.gy/kk, -(0.0+1.0j)*self.gx/kk
            self.gz = np.zeros(self.gx.shape, dtype="complex")

        del kk

        self.gx[np.isinf(self.gx)] = 0.0
        self.gx[np.isnan(self.gx)] = 0.0
        self.gy[np.isinf(self.gy)] = 0.0
        self.gy[np.isnan(self.gy)] = 0.0

        self._rot_3d(2, -beta)
        self._rot_3d(3, -alpha)

        self.gx = np.fft.ifftn(self.gx).real
        self.gy = np.fft.ifftn(self.gy).real
        self.gz = np.fft.ifftn(self.gz).real

    def __getitem__(self, item):
        if item in "xyz":
            return YTArray(getattr(self, item), "kpc")
        elif item in self.comps:
            comp = "g{}".format(item[-1])
            return YTArray(getattr(self, comp), self.units)
        else:
            raise KeyError

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
        all_comps = ["x", "y", "z"] + self.comps
        for field in all_comps:
            if in_cgs:
                self[field].in_cgs().write_hdf5(filename, dataset_name=field)
            else:
                self[field].write_hdf5(filename, dataset_name=field)
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
                                           self[self._name+"_"+ax], bounds_error=False)
            v = YTArray(func(cluster_particles[ptype, "particle_position"].d),
                        self.units)
            cluster_particles.set_field(ptype, "particle_%s_%s" % (self._name, ax), v,
                                        units=units)


class GaussianRandomField(ClusterField):
    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 alpha=-11./3., g_rms=1.0, ctr1=None, ctr2=None, r1=None,
                 r2=None, g1=None, g2=None, vector_potential=False,
                 divergence_clean=False, prng=np.random):

        super(GaussianRandomField, self).__init__(left_edge, right_edge, ddims, 
                                                  vector_potential=vector_potential,
                                                  divergence_clean=divergence_clean)

        nx, ny, nz = self.ddims

        num_halos = 0
        if r1 is not None:
            num_halos += 1
        if r2 is not None:
            num_halos += 2

        if num_halos >= 1:
            if ctr1 is None:
                ctr1 = 0.5*(self.left_edge+self.right_edge)
            else:
                ctr1 = parse_value(ctr1, "kpc").v
            r1 = parse_value(r1, "kpc").v
            g1 = parse_value(g1, self._units)
        if num_halos == 2:
            if ctr2 is None:
                raise RuntimeError("Need to specify 'ctr2' for the second halo!")
            ctr2 = parse_value(ctr2, "kpc").v
            r2 = parse_value(r2, "kpc").v
            g2 = parse_value(g2, self._units)

        # Derived stuff

        l_min = parse_value(l_min, "kpc").v
        l_max = parse_value(l_max, "kpc").v

        k0 = 2.*np.pi/l_min
        k1 = 2.*np.pi/l_max

        mylog.info("Setting up the Gaussian random fields.")

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

        kx, ky, kz = self._compute_waves()
        kk = np.sqrt(kx**2+ky**2+kz**2)
        with np.errstate(invalid='ignore', divide='ignore'):
            sigma = (1.0+(kk/k1)**2)**(0.25*alpha)*np.exp(-(kk/k0)**2)
        sigma[np.isinf(sigma)] = 0.0
        sigma[np.isnan(sigma)] = 0.0
        del kk

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

        x, y, z = self._compute_coords()

        if num_halos == 0:
            g_rms = parse_value(g_rms, self._units)
            mylog.info("Scaling the fields by the constant value %s." % g_rms)
        else:
            if num_halos == 1:
                mylog.info("Scaling the fields by cluster 1.")
                rr1 = np.sqrt((x-ctr1[0])**2 + (y-ctr1[1])**2 + (z-ctr1[2])**2)
                idxs1 = np.searchsorted(r1, rr1) - 1
                dr1 = (rr1-r1[idxs1])/(r1[idxs1+1]-r1[idxs1])
                g_rms = ((1.-dr1)*g1[idxs1] + dr1*g1[idxs1+1])**2
            if num_halos == 2:
                mylog.info("Scaling the fields by cluster 2.")
                rr2 = np.sqrt((x-ctr2[0])**2 + (y-ctr2[1])**2 + (z-ctr2[2])**2)
                idxs2 = np.searchsorted(r2, rr2) - 1
                dr2 = (rr2-r2[idxs2])/(r2[idxs2+1]-r2[idxs2])
                g_rms += ((1.-dr2)*g2[idxs2] + dr2*g2[idxs2+1])**2
            g_rms = np.sqrt(g_rms).in_units(self._units).d

        gx *= g_rms
        gy *= g_rms
        gz *= g_rms

        self.x = x[:,0,0]
        self.y = y[0,:,0]
        self.z = z[0,0,:]

        del x, y, z, g_rms

        if self.divergence_clean:
            self._divergence_clean(kx, ky, kz)

        if self.vector_potential:
            self._compute_vector_potential(kx, ky, kz)


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


class TangentialMagneticField(ClusterField):
    _units = "gauss"
    _name = "magnetic_field"
    _vector_potential = False

    def __init__(self, left_edge, right_edge, ddims, B_mag, ctr=None, radius=None):
        super(TangentialMagneticField, self).__init__(left_edge, right_edge, ddims,
                                                      vector_potential=self._vector_potential,
                                                      divergence_clean=True)
        if ctr is None:
            ctr = 0.5*(self.left_edge+self.right_edge)
        else:
            ctr = parse_value(ctr, "kpc").v

        x, y, z = self._compute_coords()

        r = np.sqrt((x-ctr[0])**2+(y-ctr[1])**2+(z-ctr[2])**2)

        self.x = x[:,0,0]
        self.y = y[0,:,0]
        self.z = z[0,0,:]

        del x, y, z

        sin_theta = np.sqrt((x-ctr[0])**2+(y-ctr[1])**2)/r
        cos_theta = (z-ctr[2])/r
        sin_phi = (y-ctr[1])/(r*sin_theta)
        cos_phi = (x-ctr[0])/(r*sin_theta)

        g_theta = sin_theta*(1.0-sin_phi**2)
        g_phi = -8.0*sin_theta*cos_theta*sin_phi*cos_phi

        self.gx = B_mag*(g_theta*cos_theta*cos_phi-g_phi*sin_phi)
        self.gy = B_mag*(g_theta*cos_theta*sin_phi+g_phi*cos_phi)
        self.gz = -B_mag*g_theta*sin_theta

        if radius is not None:
            radius = parse_value(radius, "kpc").v
            self.gx[r > radius] = 0.0
            self.gy[r > radius] = 0.0
            self.gz[r > radius] = 0.0

        kx, ky, kz = self._compute_waves()

        self._divergence_clean(kx, ky, kz)

        if self.vector_potential:
            self._compute_vector_potential(kx, ky, kz)


class RandomMagneticVectorPotential(RandomMagneticField):
    _name = "magnetic_vector_potential"
    _vector_potential = True


class TangentialMagneticVectorPotential(TangentialMagneticField):
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
