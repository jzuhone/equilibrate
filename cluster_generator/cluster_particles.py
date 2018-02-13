from yt import mylog, YTArray, uconcatenate, load_particles
from collections import OrderedDict, defaultdict
from yt.funcs import ensure_list, ensure_numpy_array
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py
import numpy as np
import os

gadget_fields = {"dm": ["Coordinates", "Velocities", "Masses", "ParticleIDs"],
                 "gas": ["Coordinates", "Velocities", "Masses", "ParticleIDs",
                         "InternalEnergy", "MagneticField","Density",
                         "MagneticVectorPotential"],
                 "star": ["Coordinates", "Velocities", "Masses", "ParticleIDs"]}

gadget_field_map = {"Coordinates": "particle_position",
                    "Velocities": "particle_velocity",
                    "Masses": "particle_mass",
                    "Density": "particle_density",
                    "InternalEnergy": "particle_thermal_energy",
                    "MagneticField": "particle_magnetic_field",
                    "MagneticVectorPotential": "particle_magnetic_vector_potential"}

gadget_field_units = {"Coordinates": "kpc",
                      "Velocities": "km/s",
                      "Masses": "1e10*Msun",
                      "Density": "1e10*Msun/kpc**3",
                      "InternalEnergy": "km**2/s**2",
                      "MagneticField": "gauss",
                      "MagneticVectorPotential": "gauss*kpc"}

ptype_map = OrderedDict([("PartType0", "gas"),
                         ("PartType1", "dm"),
                         ("PartType4", "star")])

rptype_map = OrderedDict([(v, k) for k, v in ptype_map.items()])

class ClusterParticles(object):
    def __init__(self, particle_types, fields):
        self.particle_types = ensure_list(particle_types)
        self.fields = fields
        self._update_num_particles()
        self.field_names = defaultdict(list)
        for field in self.fields:
            self.field_names[field[0]].append(field[1])

    @classmethod
    def from_h5_file(cls, filename, ptypes=None):
        r"""
        Generate cluster particles from an HDF5 file.

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.

        Examples
        --------
        >>> from cluster_generator import ClusterParticles
        >>> dm_particles = ClusterParticles.from_h5_file("dm_particles.h5")
        """
        names = {}
        f = h5py.File(filename)
        if ptypes is None:
            ptypes = list(f.keys())
        for ptype in ptypes:
            names[ptype] = list(f[ptype].keys())
        f.close()

        fields = OrderedDict()
        for ptype in ptypes:
            for field in names[ptype]:
                fields[ptype, field] = YTArray.from_hdf5(filename, dataset_name=field,
                                                         group_name=ptype).in_base("galactic")
        return cls(ptypes, fields)

    @classmethod
    def from_gadget_ics(cls, filename, ptypes=None):
        fields = OrderedDict()
        f = h5py.File(filename, "r")
        particle_types = []
        if ptypes is None:
            ptypes = [k for k in f if k.startswith("PartType")]
        for ptype in ptypes:
            my_ptype = ptype_map[ptype]
            particle_types.append(my_ptype)
            g = f[ptype]
            for field in gadget_fields[my_ptype]:
                if field in g:
                    if field != "ParticleIDs":
                        fd = gadget_field_map[field]
                        units = gadget_field_units[field]
                        fields[my_ptype, fd] = YTArray(g[field], units).in_base("galactic")
        f.close()
        return cls(particle_types, fields)

    def _update_num_particles(self):
        self.num_particles = {}
        for ptype in self.particle_types:
            self.num_particles[ptype] = self.fields[ptype, "particle_mass"].size

    def pick_stars_from_dm(self, nstars, pick_bcg=True):
        idxs = np.random.random_integers(0, self.num_particles["dm"], size=nstars)
        if pick_bcg:
            if "particle_energy" not in self.fields:
                raise RuntimeError("Cannot determine particle energy!")
            idx = np.argmin(self.fields["dm","particle_energy"])
            idxs = np.append(idxs, idx)
        mask = np.zeros(self.num_particles["dm"], dtype=bool)
        mask[idxs] = True
        for field in self.field_names["dm"]:
            if "star" not in self.particle_types:
                self.fields["star", field] = self.fields["dm", field][mask]
            else:
                self.fields["star", field] = uconcatenate(self.fields["star", field],
                                                          self.fields["dm", field][mask])
            self.fields["dm", field] = self.fields["dm", field][idxs]
        if "star" not in self.particle_types:
            self.particle_types.append("star")
        self._update_num_particles()

    def make_radial_cut(self, r_max, p_type="all"):
        if p_type == "all":
            chop_types = self.particle_types
        else:
            chop_types = [p_type]
        for pt in chop_types:
            cidx = np.sqrt((self[pt, "particle_position"]**2).sum(axis=1)).d <= r_max
            for field in self.field_names[pt]:
                self.fields[pt, field] = self.fields[pt, field][cidx]
        self._update_num_particles()

    def write_particles_to_h5(self, output_filename, in_cgs=False, overwrite=False):
        """
        Write the particles to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the particles to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % output_filename)
        f = h5py.File(output_filename, "w")
        [f.create_group(ptype) for ptype in self.particle_types]
        f.flush()
        f.close()
        for field in self.fields:
            if in_cgs:
                fd = self.fields[field].in_cgs()
            else:
                fd = self.fields[field]
            fd.write_hdf5(output_filename, dataset_name=field[1],
                          group_name=field[0])

    def __add__(self, other):
        fields = self.fields.copy()
        for field in other.fields:
            if field in fields:
                fields[field] = uconcatenate([self[field], other[field]])
            else:
                fields[field] = other[field]
        particle_types = list(set(self.particle_types + other.particle_types))
        return ClusterParticles(particle_types, fields)

    def add_offsets(self, r_ctr, v_ctr):
        if not isinstance(r_ctr, YTArray):
            r_ctr = YTArray(r_ctr, "kpc")
        if not isinstance(v_ctr, YTArray):
            v_ctr = YTArray(v_ctr, "kpc/Myr")
        for ptype in self.particle_types:
            self.fields[ptype, "particle_position"] += r_ctr
            self.fields[ptype, "particle_velocity"] += v_ctr

    def _clip_to_box(self, ptype, box_size):
        pos = self.fields[ptype, "particle_position"]
        return ~np.logical_or((pos < 0.0).any(axis=1), (pos > box_size).any(axis=1))

    def _write_gadget_fields(self, ptype, h5_group, idxs, dtype):
        for field in gadget_fields[ptype]:
            if field == "ParticleIDs":
                continue
            my_field = gadget_field_map[field]
            if (ptype, my_field) in self.fields:
                units = gadget_field_units[field]
                data = self.fields[ptype, my_field][idxs].in_units(units).d.astype(dtype)
                h5_group.create_dataset(field, data=data)

    def write_to_gadget_ics(self, ic_filename, box_size,
                            dtype='float32', overwrite=False):
        if os.path.exists(ic_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % ic_filename)
        num_particles = {}
        npart = 0
        mass_table = np.zeros(6)
        f = h5py.File(ic_filename, "w")
        for ptype in self.particle_types:
            gptype = rptype_map[ptype]
            idxs = self._clip_to_box(ptype, box_size)
            num_particles[ptype] = idxs.sum()
            g = f.create_group(gptype)
            self._write_gadget_fields(ptype, g, idxs, dtype)
            ids = np.arange(num_particles[ptype])+1+npart
            g.create_dataset("ParticleIDs", data=ids.astype("uint32"))
            npart += num_particles[ptype]
            if ptype in ["star", "dm"]:
                mass_table[int(rptype_map[ptype][-1])] = g["Masses"][0]
        f.flush()
        hg = f.create_group("Header")
        hg.attrs["Time"] = 0.0
        hg.attrs["Redshift"] = 0.0
        hg.attrs["BoxSize"] = box_size
        hg.attrs["Omega0"] = 0.0
        hg.attrs["OmegaLambda"] = 0.0
        hg.attrs["HubbleParam"] = 1.0
        hg.attrs["NumPart_ThisFile"] = np.array([num_particles.get("gas", 0),
                                                 num_particles.get("dm", 0),
                                                 0, 0,
                                                 num_particles.get("star", 0),
                                                 0], dtype='uint32')
        hg.attrs["NumPart_Total"] = hg.attrs["NumPart_ThisFile"]
        hg.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype='uint32')
        hg.attrs["NumFilesPerSnapshot"] = 1
        hg.attrs["MassTable"] = mass_table
        hg.attrs["Flag_Sfr"] = 0
        hg.attrs["Flag_Cooling"] = 0
        hg.attrs["Flag_StellarAge"] = 0
        hg.attrs["Flag_Metals"] = 0
        hg.attrs["Flag_Feedback"] = 0
        hg.attrs["Flag_DoublePrecision"] = 0
        hg.attrs["Flag_IC_Info"] = 0
        f.flush()
        f.close()

    def set_field(self, ptype, name, value, units=None):
        """
        Set a field with name *name* to value *value*, which is a YTArray.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, YTArray):
            raise TypeError("value needs to be a YTArray")
        num_particles = self.num_particles[ptype]
        if value.size == num_particles:
            if (ptype, name) in self.fields:
                mylog.warning("Overwriting field (%s, %s)." % (ptype, name))
            self.fields[ptype, name] = value
            if units is not None:
                self.fields[ptype, name].convert_to_units(units)
        else:
            raise ValueError("The length of the array needs to be %d particles!"
                             % num_particles)

    def __getitem__(self, key):
        return self.fields[key]

    def __setitem__(self, key, value):
        self.fields[key] = value

    def keys(self):
        return self.fields.keys()

    def to_yt_dataset(self, box_size):
        data = self.fields.copy()
        for ptype in self.particle_types:
            pos = data.pop((ptype, "particle_position"))
            vel = data.pop((ptype, "particle_velocity"))
            for i, ax in enumerate("xyz"):
                data[ptype,"particle_position_%s" % ax] = pos[:,i]
                data[ptype,"particle_velocity_%s" % ax] = vel[:,i]
        return load_particles(data, length_unit="kpc", bbox=[[0.0, box_size]]*3,
                              mass_unit="Msun", time_unit="Myr")


def combine_two_clusters(particles1, particles2, hse1, hse2,
                         center1, center2, velocity1, velocity2, box_size):
    center1 = ensure_numpy_array(center1)
    center2 = ensure_numpy_array(center2)
    particles1.add_offsets(center1, [0.0]*3)
    particles2.add_offsets(center2, [0.0]*3)
    particles = particles1+particles2
    r1 = ((particles["gas", "particle_position"].d-center1)**2).sum(axis=1)
    np.sqrt(r1, r1)
    r2 = ((particles["gas", "particle_position"].d-center2)**2).sum(axis=1)
    np.sqrt(r2, r2)
    get_density1 = InterpolatedUnivariateSpline(hse1["radius"], hse1["density"])
    dens1 = get_density1(r1)
    get_density2 = InterpolatedUnivariateSpline(hse2["radius"], hse2["density"])
    dens2 = get_density2(r2)
    dens = dens1+dens2
    e_arr1 = 1.5*hse1["pressure"]/hse1["density"]
    get_energy1 = InterpolatedUnivariateSpline(hse1["radius"], e_arr1)
    eint1 = get_energy1(r1)
    e_arr2 = 1.5*hse2["pressure"]/hse2["density"]
    get_energy2 = InterpolatedUnivariateSpline(hse2["radius"], e_arr2)
    eint2 = get_energy2(r2)
    particles["gas", "particle_density"] = YTArray(dens, "Msun/kpc**3")
    particles["gas", "particle_thermal_energy"] = YTArray((eint1*dens1+eint2*dens2)/dens,
                                                          "kpc**2/Myr**2")
    particles["gas", "particle_velocity"] = YTArray((velocity1*dens1+velocity2*dens2)/dens,
                                                    "kpc/Myr")
    return particles
