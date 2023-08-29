"""
3D fields for magnetic field initiation and other field based tasks.
"""
import os

import numpy as np
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog, parse_prng


def parse_value(value, default_units):
    """
    Parses an array of values into the correct units.
    Parameters
    ----------
    value: array-like or tuple
        The array from which to convert values to correct units. If ``value`` is a ``unyt_array``, the unit is simply converted,
        if ``value`` is a tuple in the form ``(v_array,v_unit)``, the conversion will be made and will return an ``unyt_array``.
        Finally, if ``value`` is an array, it is assumed that the ``default_units`` are correct.
    default_units: str
        The default unit for the quantity.

    Returns
    -------
    unyt_array:
        The converted array.
    """
    if isinstance(value, unyt_array):
        val = unyt_array(value.v, value.units).in_units(default_units)
    elif isinstance(value, tuple):
        val = unyt_array(value[0], value[1]).in_units(default_units)
    else:
        val = unyt_array(value, default_units)
    return val


def rot_3d(axis, gx, gy, gz, ang):
    """
    Rotates the vector ``[gx,gy,gz]`` by an angle ``ang`` around a specified axis.

    Parameters
    ----------
    axis: int
        The axis to rotate about. Options are ``1,2,3``.
    gx: float
    gy: float
    gz: float
    ang: float
        The angle over which to rotate.

    Returns
    -------
    gx: float
    gy: float
    gz: float
    """

    c = np.cos(ang)
    s = np.sin(ang)

    if axis == 1:
        gy, gz = c * gy + s * gz, -s * gy + c * gz
    elif axis == 2:
        gx, gz = c * gx - s * gz, s * gx + c * gz
    elif axis == 3:
        gx, gy = c * gx + s * gy, -s * gx + c * gy

    return gx, gy, gz


class ClusterField:
    _units = "dimensionless"
    _name = "vector"

    def __init__(self, left_edge, right_edge, ddims, padding=0.1,
                 vector_potential=False, divergence_clean=False):
        """

        Parameters
        ----------
        left_edge: array-like
            The lower edge of the box [kpc] for each of the dimensions.
        right_edge: array-like
            The upper edge of the box [kpc] for each of the dimensions.
        ddims: array-like
            The number of grids in each of the axes.
        padding: The amount of additional padding to add to the boundary.
        vector_potential
        divergence_clean
        """
        #  Basic Management
        # ----------------------------------------------------------------------------------------------------------------- #
        # - Type conversions and setup
        ddims = np.array(ddims).astype("int")
        left_edge = parse_value(left_edge, "kpc").v
        right_edge = parse_value(right_edge, "kpc").v
        width = right_edge - left_edge
        deltas = width / ddims

        # - Accounting for padding - #
        pad_dims = (2 * np.ceil(0.5 * padding * ddims)).astype("int")
        self.left_edge = left_edge - 0.5 * pad_dims * deltas
        self.right_edge = right_edge + 0.5 * pad_dims * deltas
        self.ddims = ddims + pad_dims

        # - Additional attributes - #
        self.vector_potential = vector_potential
        self.divergence_clean = divergence_clean
        self.dx, self.dy, self.dz = deltas
        self.comps = [f"{self._name}_{ax}" for ax in "xyz"]

    def _compute_coords(self):
        """Creates the coordinate mgrids from the grid"""
        nx, ny, nz = self.ddims
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz] + 0.5
        x *= self.dx
        y *= self.dy
        z *= self.dz
        x += self.left_edge[0]
        y += self.left_edge[1]
        z += self.left_edge[2]
        return x, y, z

    def _compute_waves(self):
        nx, ny, nz = self.ddims

        kx, ky, kz = np.mgrid[0:nx, 0:ny, 0:nz].astype("float64")
        kx[kx > nx // 2] = kx[kx > nx // 2] - nx
        ky[ky > ny // 2] = ky[ky > ny // 2] - ny
        kz[kz > nz // 2] = kz[kz > nz // 2] - nz
        kx *= 2. * np.pi / (nx * self.dx)
        ky *= 2. * np.pi / (ny * self.dy)
        kz *= 2. * np.pi / (nz * self.dz)

        return kx, ky, kz

    def _rot_3d(self, axis, ang):
        c = np.cos(ang)
        s = np.sin(ang)

        if axis == 1:
            self.gy, self.gz = c * self.gy + s * self.gz, -s * self.gy + c * self.gz
        elif axis == 2:
            self.gx, self.gz = c * self.gx - s * self.gz, s * self.gx + c * self.gz
        elif axis == 3:
            self.gx, self.gy = c * self.gx + s * self.gy, -s * self.gx + c * self.gy

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
            [(1.0 - kxd * kxd) * self.gx - kxd * kyd * self.gy - kxd * kzd * self.gz,
             -kyd * kxd * self.gx + (1.0 - kyd * kyd) * self.gy - kyd * kzd * self.gz,
             -kzd * kxd * self.gx - kzd * kyd * self.gy + (1.0 - kzd * kzd) * self.gz]

        self.gx = np.fft.ifftn(self.gx).real
        self.gy = np.fft.ifftn(self.gy).real
        self.gz = np.fft.ifftn(self.gz).real

        del kxd, kyd, kzd, kkd

    def _compute_vector_potential(self, kx, ky, kz):

        kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

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
            self.gx, self.gy = (0.0 + 1.0j) * self.gy / kk, -(0.0 + 1.0j) * self.gx / kk
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
            return unyt_array(getattr(self, item), "kpc")
        elif item in self.comps:
            comp = f"g{item[-1]}"
            return unyt_array(getattr(self, comp), self.units)
        else:
            raise KeyError

    @property
    def units(self):
        if self.vector_potential:
            return f"{self._units}*kpc"
        else:
            return self._units

    def write_file(self, filename, overwrite=False,
                   length_unit=None, field_unit=None,
                   format='hdf5'):
        r"""
        Write the 3D field to a file. The coordinates of
        the cells along the different axes are also written.

        Parameters
        ----------
        filename : string
            The name of the file to write the fields to.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        length_unit : string, optional
            The length unit (affects coordinates and potential fields).
            Default: "kpc"
        field_unit : string, optional
            The units for the field 
        """
        import h5py
        from scipy.io import FortranFile
        if length_unit is None:
            length_unit = "kpc"
        if os.path.exists(filename) and not overwrite:
            raise IOError(f"Cannot create {filename}. "
                          f"It exists and overwrite=False.")
        all_comps = ["x", "y", "z"] + self.comps
        if format == "hdf5":
            write_class = h5py.File
        elif format == "fortran":
            write_class = FortranFile
        with write_class(filename, "w") as f:
            if format == "fortran":
                f.write_record(self["x"].size)
            for field in all_comps:
                if field in "xyz":
                    fd = self[field].to(length_unit)
                elif field_unit is not None:
                    if self.vector_potential:
                        units = f"{length_unit}*{field_unit}"
                    else:
                        units = field_unit
                    fd = self[field].to(units)
                else:
                    fd = self[field]
                if format == "hdf5":
                    d = f.create_dataset(field, data=fd.d)
                    d.attrs["units"] = str(fd.units)
                elif format == "fortran":
                    f.write_record(fd.d)
            if format == "hdf5":
                f.attrs["name"] = self._name
                f.attrs["units"] = self.units
                f.attrs["vector_potential"] = int(self.vector_potential)

    def map_field_to_particles(self, cluster_particles,
                               ptype="gas", units=None):
        r"""
        Map the 3D field to a set of particles, creating new
        particle fields. This uses tri-linear interpolation. 

        Parameters
        ----------
        cluster_particles : :class:`~cluster_generator.particles.ClusterParticles`
            The ClusterParticles object which will have new
            fields added.
        ptype : string, optional
            The particle type to add the new fields to. Default:
            "gas", which will almost always be the case.
        units : string, optional
            Change the units of the field. Default: None, which
            implies they will remain in "galactic" units. 
        """
        from scipy.interpolate import RegularGridInterpolator
        v = np.zeros((cluster_particles.num_particles[ptype], 3))
        for i, ax in enumerate("xyz"):
            func = RegularGridInterpolator((self["x"], self["y"], self["z"]),
                                           self[self._name + "_" + ax],
                                           bounds_error=False, fill_value=0.0)
            v[:, i] = func(cluster_particles[ptype, "particle_position"].d)
        cluster_particles.set_field(ptype, self._name,
                                    unyt_array(v, self.units), units=units)


class GaussianRandomField(ClusterField):
    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 padding=0.1, alpha=-11. / 3., g_rms=1.0, ctr1=None,
                 ctr2=None, ctr3=None, r1=None, r2=None, r3=None,
                 g1=None, g2=None, g3=None, vector_potential=False,
                 divergence_clean=False, prng=None, r_max=None):

        prng = parse_prng(prng)

        super(GaussianRandomField, self).__init__(left_edge, right_edge, ddims,
                                                  padding=padding,
                                                  vector_potential=vector_potential,
                                                  divergence_clean=divergence_clean)

        nx, ny, nz = self.ddims

        num_halos = 0
        if r1 is not None:
            num_halos += 1
        if r2 is not None:
            num_halos += 1
        if r3 is not None:
            num_halos += 1

        if num_halos >= 1:
            if ctr1 is None:
                ctr1 = 0.5 * (self.left_edge + self.right_edge)
            else:
                ctr1 = parse_value(ctr1, "kpc").v
            r1 = parse_value(r1, "kpc").v
            g1 = parse_value(g1, self._units)
        if num_halos >= 2:
            if ctr2 is None:
                raise RuntimeError("Need to specify 'ctr2' for the second halo!")
            ctr2 = parse_value(ctr2, "kpc").v
            r2 = parse_value(r2, "kpc").v
            g2 = parse_value(g2, self._units)
        if num_halos == 3:
            if ctr3 is None:
                raise RuntimeError("Need to specify 'ctr3' for the second halo!")
            ctr3 = parse_value(ctr3, "kpc").v
            r3 = parse_value(r3, "kpc").v
            g3 = parse_value(g3, self._units)

        # Derived stuff

        l_min = parse_value(l_min, "kpc").v
        l_max = parse_value(l_max, "kpc").v

        k0 = 2. * np.pi / l_min
        k1 = 2. * np.pi / l_max

        mylog.info("Setting up the Gaussian random fields.")

        v = np.exp(2. * np.pi * 1j * prng.random((3, nx, ny, nz)))

        v[:, 0, 0, 0] = 2. * np.sign((v[:, 0, 0, 0].imag < 0.0).astype("int")) - 1. + 0j
        v[:, nx // 2, ny // 2, nz // 2] = 2. * np.sign(
            (v[:, nx // 2, ny // 2, nz // 2].imag < 0.0).astype("int")) - 1. + 0j
        v[:, 0, ny // 2, nz // 2] = 2. * np.sign((v[:, 0, ny // 2, nz // 2].imag < 0.0).astype("int")) - 1. + 0j
        v[:, nx // 2, 0, nz // 2] = 2. * np.sign((v[:, nx // 2, 0, nz // 2].imag < 0.0).astype("int")) - 1. + 0j
        v[:, nx // 2, ny // 2, 0] = 2. * np.sign((v[:, nx // 2, ny // 2, 0].imag < np.pi).astype("int")) - 1. + 0j
        v[:, 0, 0, nz // 2] = 2. * np.sign((v[:, 0, 0, nz // 2].imag < np.pi).astype("int")) - 1. + 0j
        v[:, 0, ny // 2, 0] = 2. * np.sign((v[:, 0, ny // 2, 0].imag < np.pi).astype("int")) - 1. + 0j
        v[:, nx // 2, 0, 0] = 2. * np.sign((v[:, nx // 2, 0, 0].imag < np.pi).astype("int")) - 1. + 0j

        np.multiply(v, np.sqrt(-2. * np.log(prng.random((3, nx, ny, nz)))), v)

        kx, ky, kz = self._compute_waves()
        kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            sigma = (1.0 + (kk / k1) ** 2) ** (0.25 * alpha) * np.exp(-0.5 * (kk / k0) ** 2)
        sigma[np.isinf(sigma)] = 0.0
        sigma[np.isnan(sigma)] = 0.0
        del kk

        v[:, nx - 1:0:-1, ny - 1:0:-1, nz - 1:nz // 2:-1] = np.conjugate(v[:, 1:nx, 1:ny, 1:nz // 2])
        v[:, nx - 1:0:-1, ny - 1:ny // 2:-1, nz // 2] = np.conjugate(v[:, 1:nx, 1:ny // 2, nz // 2])
        v[:, nx - 1:0:-1, ny - 1:ny // 2:-1, 0] = np.conjugate(v[:, 1:nx, 1:ny // 2, 0])
        v[:, nx - 1:0:-1, 0, nz - 1:nz // 2:-1] = np.conjugate(v[:, 1:nx, 0, 1:nz // 2])
        v[:, 0, ny - 1:0:-1, nz - 1:nz // 2:-1] = np.conjugate(v[:, 0, 1:ny, 1:nz // 2])
        v[:, nx - 1:nx // 2:-1, ny // 2, nz // 2] = np.conjugate(v[:, 1:nx // 2, ny // 2, nz // 2])
        v[:, nx - 1:nx // 2:-1, ny // 2, 0] = np.conjugate(v[:, 1:nx // 2, ny // 2, 0])
        v[:, nx - 1:nx // 2:-1, 0, nz // 2] = np.conjugate(v[:, 1:nx // 2, 0, nz // 2])
        v[:, 0, ny - 1:ny // 2:-1, nz // 2] = np.conjugate(v[:, 0, 1:ny // 2, nz // 2])
        v[:, nx - 1:nx // 2:-1, 0, 0] = np.conjugate(v[:, 1:nx // 2, 0, 0])
        v[:, 0, ny - 1:ny // 2:-1, 0] = np.conjugate(v[:, 0, 1:ny // 2, 0])
        v[:, 0, 0, nz - 1:nz // 2:-1] = np.conjugate(v[:, 0, 0, 1:nz // 2])

        gx = np.fft.ifftn(sigma * v[0, :, :, :]).real
        gy = np.fft.ifftn(sigma * v[1, :, :, :]).real
        gz = np.fft.ifftn(sigma * v[2, :, :, :]).real

        del sigma, v

        g_avg = np.sqrt(np.mean(gx * gx + gy * gy + gz * gz))

        gx /= g_avg
        gy /= g_avg
        gz /= g_avg

        x, y, z = self._compute_coords()

        if num_halos == 0:
            g_rms = parse_value(g_rms, self._units)
            mylog.info(f"Scaling the fields by the constant value {g_rms}.")
        else:
            if num_halos >= 1:
                mylog.info("Scaling the fields by cluster 1.")
                rr1 = np.sqrt((x - ctr1[0]) ** 2 + (y - ctr1[1]) ** 2 + (z - ctr1[2]) ** 2)
                if r_max is not None:
                    rr1[rr1 > r_max] = r_max
                idxs1 = np.searchsorted(r1, rr1) - 1
                dr1 = (rr1 - r1[idxs1]) / (r1[idxs1 + 1] - r1[idxs1])
                g_rms = ((1. - dr1) * g1[idxs1] + dr1 * g1[idxs1 + 1]) ** 2
            if num_halos >= 2:
                mylog.info("Scaling the fields by cluster 2.")
                rr2 = np.sqrt((x - ctr2[0]) ** 2 + (y - ctr2[1]) ** 2 + (z - ctr2[2]) ** 2)
                if r_max is not None:
                    rr2[rr2 > r_max] = r_max
                idxs2 = np.searchsorted(r2, rr2) - 1
                dr2 = (rr2 - r2[idxs2]) / (r2[idxs2 + 1] - r2[idxs2])
                g_rms += ((1. - dr2) * g2[idxs2] + dr2 * g2[idxs2 + 1]) ** 2
            if num_halos == 3:
                mylog.info("Scaling the fields by cluster 3.")
                rr3 = np.sqrt((x - ctr3[0]) ** 2 + (y - ctr3[1]) ** 2 + (z - ctr3[2]) ** 2)
                if r_max is not None:
                    rr3[rr3 > r_max] = r_max
                idxs3 = np.searchsorted(r3, rr3) - 1
                dr3 = (rr3 - r3[idxs3]) / (r3[idxs3 + 1] - r3[idxs3])
                g_rms += ((1. - dr3) * g3[idxs3] + dr3 * g3[idxs3 + 1]) ** 2
            g_rms = np.sqrt(g_rms).in_units(self._units).d

        gx *= g_rms
        gy *= g_rms
        gz *= g_rms

        self.x = x[:, 0, 0]
        self.y = y[0, :, 0]
        self.z = z[0, 0, :]

        self.gx = gx
        self.gy = gy
        self.gz = gz

        rescale = (self.gx ** 2 + self.gy ** 2 + self.gz ** 2).sum()

        del x, y, z, g_rms

        if self.divergence_clean:
            self._divergence_clean(kx, ky, kz)

        rescale /= (self.gx ** 2 + self.gy ** 2 + self.gz ** 2).sum()

        self.gx *= rescale
        self.gy *= rescale
        self.gz *= rescale

        if self.vector_potential:
            self._compute_vector_potential(kx, ky, kz)


class RandomMagneticField(GaussianRandomField):
    _units = "gauss"
    _name = "magnetic_field"
    _vector_potential = False

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 B_rms, padding=0.1, alpha=-11. / 3., prng=None):
        super(RandomMagneticField, self).__init__(left_edge, right_edge, ddims,
                                                  l_min, l_max, padding=padding, alpha=alpha, divergence_clean=True,
                                                  g_rms=B_rms, vector_potential=self._vector_potential, prng=prng)


class RadialRandomMagneticField(GaussianRandomField):
    _units = "gauss"
    _name = "magnetic_field"
    _vector_potential = False

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 ctr1, profile1, padding=0.1, ctr2=None, profile2=None,
                 ctr3=None, profile3=None, alpha=-11. / 3., r_max=None,
                 prng=None):
        if isinstance(profile1, ClusterModel):
            r1 = profile1["radius"].to_value("kpc")
            B1 = profile1["magnetic_field_strength"]
        elif isinstance(profile1, str):
            r1 = unyt_array.from_hdf5(profile1, dataset_name="radius",
                                      group_name="fields").to('kpc').d
            B1 = unyt_array.from_hdf5(profile1,
                                      dataset_name="magnetic_field_strength", group_name="fields")
        else:
            r1, B1 = profile1
        if profile2 is not None:
            if isinstance(profile2, ClusterModel):
                r2 = profile2["radius"].to_value("kpc")
                B2 = profile2["magnetic_field_strength"]
            elif isinstance(profile2, str):
                r2 = unyt_array.from_hdf5(profile2, dataset_name="radius",
                                          group_name="fields").to('kpc').d
                B2 = unyt_array.from_hdf5(profile2,
                                          dataset_name="magnetic_field_strength", group_name="fields")
            else:
                r2, B2 = profile2
        else:
            r2 = None
            B2 = None
        if profile3 is not None:
            if isinstance(profile3, ClusterModel):
                r3 = profile3["radius"].to_value("kpc")
                B3 = profile3["magnetic_field_strength"]
            elif isinstance(profile3, str):
                r3 = unyt_array.from_hdf5(profile3, dataset_name="radius",
                                          group_name="fields").to('kpc').d
                B3 = unyt_array.from_hdf5(profile3,
                                          dataset_name="magnetic_field_strength", group_name="fields")
            else:
                r3, B3 = profile3
        else:
            r3 = None
            B3 = None
        super(RadialRandomMagneticField, self).__init__(
            left_edge, right_edge, ddims, l_min, l_max, padding=padding,
            alpha=alpha, ctr1=ctr1, ctr2=ctr2, ctr3=ctr3, r1=r1, r2=r2,
            r3=r3, g1=B1, g2=B2, g3=B3, divergence_clean=True, r_max=r_max,
            vector_potential=self._vector_potential, prng=prng)


class RandomMagneticVectorPotential(RandomMagneticField):
    _name = "magnetic_vector_potential"
    _vector_potential = True


class RadialRandomMagneticVectorPotential(RadialRandomMagneticField):
    _name = "magnetic_vector_potential"
    _vector_potential = True


class RandomVelocityField(GaussianRandomField):
    _units = "kpc/Myr"
    _name = "velocity"

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 V_rms, padding=0.1, alpha=-11. / 3., divergence_clean=False,
                 prng=None):
        super(RandomVelocityField, self).__init__(left_edge, right_edge, ddims,
                                                  l_min, l_max, padding=padding, g_rms=V_rms, alpha=alpha, prng=prng,
                                                  divergence_clean=divergence_clean)


class RadialRandomVelocityField(GaussianRandomField):
    _units = "kpc/Myr"
    _name = "velocity"

    def __init__(self, left_edge, right_edge, ddims, l_min, l_max,
                 ctr1, profile1, padding=0.1, ctr2=None, profile2=None,
                 ctr3=None, profile3=None, alpha=-11. / 3., r_max=None,
                 divergence_clean=False, prng=None):
        if isinstance(profile1, ClusterModel):
            r1 = profile1["radius"].to_value("kpc")
            V1 = profile1["velocity_dispersion"]
        elif isinstance(profile1, str):
            r1 = unyt_array.from_hdf5(profile1, dataset_name="radius",
                                      group_name="fields").d
            V1 = unyt_array.from_hdf5(profile1,
                                      dataset_name="velocity_dispersion", group_name="fields")
        else:
            r1, V1 = profile1
        if profile2 is not None:
            if isinstance(profile2, ClusterModel):
                r2 = profile2["radius"].to_value("kpc")
                V2 = profile2["velocity_dispersion"]
            elif isinstance(profile2, str):
                r2 = unyt_array.from_hdf5(profile2, dataset_name="radius",
                                          group_name="fields").d
                V2 = unyt_array.from_hdf5(profile2,
                                          dataset_name="velocity_dispersion", group_name="fields")
            else:
                r2, V2 = profile2
        else:
            r2 = None
            V2 = None
        if profile3 is not None:
            if isinstance(profile3, ClusterModel):
                r3 = profile3["radius"].to_value("kpc")
                V3 = profile3["velocity_dispersion"]
            elif isinstance(profile3, str):
                r3 = unyt_array.from_hdf5(profile3, dataset_name="radius",
                                          group_name="fields").to('kpc').d
                V3 = unyt_array.from_hdf5(profile3,
                                          dataset_name="velocity_dispersion", group_name="fields")
            else:
                r3, V3 = profile3
        else:
            r3 = None
            V3 = None
        super(RadialRandomVelocityField, self).__init__(left_edge, right_edge,
                                                        ddims, l_min, l_max, padding=padding, alpha=alpha, ctr1=ctr1,
                                                        ctr2=ctr2, ctr3=ctr3, r1=r1, r2=r2, r3=r3, g1=V1, g2=V2, g3=V3,
                                                        divergence_clean=divergence_clean, r_max=r_max, prng=prng)
