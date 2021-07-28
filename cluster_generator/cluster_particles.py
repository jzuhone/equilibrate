from yt import YTArray, YTQuantity, uconcatenate, load_particles
from collections import OrderedDict, defaultdict
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import ensure_ytarray, ensure_list, mylog
import h5py
import numpy as np
import os
from more_itertools import always_iterable

ensure_list = lambda x: list(always_iterable(x))

gadget_fields = {"dm": ["Coordinates", "Velocities", "Masses", "ParticleIDs",
                        "Potential"],
                 "gas": ["Coordinates", "Velocities", "Masses", "ParticleIDs",
                         "InternalEnergy", "MagneticField", "Density", "Potential",
                         "PassiveScalars"],
                 "star": ["Coordinates", "Velocities", "Masses", "ParticleIDs",
                          "Potential"],
                 "black_hole": ["Coordinates", "Velocities", "Masses", "ParticleIDs",
                                "Potential"]}

gadget_field_map = {"Coordinates": "particle_position",
                    "Velocities": "particle_velocity",
                    "Masses": "particle_mass",
                    "Density": "density",
                    "Potential": "potential_energy",
                    "InternalEnergy": "thermal_energy",
                    "MagneticField": "magnetic_field"}

gadget_field_units = {"Coordinates": "kpc",
                      "Velocities": "km/s",
                      "Masses": "1e10*Msun",
                      "Density": "1e10*Msun/kpc**3",
                      "InternalEnergy": "km**2/s**2",
                      "Potential": "km**2/s**2",
                      "PassiveScalars": "",
                      "MagneticField": "1e5*sqrt(Msun)*km/s/(kpc**1.5)"}

ptype_map = OrderedDict([("PartType0", "gas"),
                         ("PartType1", "dm"),
                         ("PartType4", "star"),
                         ("PartType5", "black_hole")])

rptype_map = OrderedDict([(v, k) for k, v in ptype_map.items()])


class ClusterParticles(object):
    def __init__(self, particle_types, fields, box_size=None):
        self.particle_types = ensure_list(particle_types)
        self.fields = fields
        self._update_num_particles()
        self._update_field_names()
        self.box_size = box_size
        self.passive_scalars = []

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
        with h5py.File(filename, "r") as f:
            if ptypes is None:
                ptypes = list(f.keys())
            ptypes = ensure_list(ptypes)
            for ptype in ptypes:
                names[ptype] = list(f[ptype].keys())
        fields = OrderedDict()
        for ptype in ptypes:
            for field in names[ptype]:
                if field == "particle_index":
                    with h5py.File(filename, "r") as f:
                        fields[ptype, field] = f[ptype][field][:]
                else:
                    a = YTArray.from_hdf5(filename, dataset_name=field,
                                          group_name=ptype)
                    fields[ptype, field] = YTArray(
                        a.d.astype("float64"), str(a.units)).in_base("galactic")
        return cls(ptypes, fields)

    @classmethod
    def from_gadget_file(cls, filename, ptypes=None):
        """
        Read in particle data from a Gadget snapshot

        Parameters
        ----------
        filename : string
            The name of the file to read from.
        ptypes : string or list of strings, optional
            The particle types to read from the file, either
            a single string or a list of strings. If None,
            all particle types will be read from the file.

        Examples
        --------
        >>> from cluster_generator import ClusterParticles
        >>> ptypes = ["gas", "dm"]
        >>> particles = ClusterParticles.from_gadget_file("snapshot_060.h5", ptypes=ptypes)
        """
        fields = OrderedDict()
        f = h5py.File(filename, "r")
        particle_types = []
        if ptypes is None:
            ptypes = [k for k in f if k.startswith("PartType")]
        else:
            ptypes = ensure_list(ptypes)
            ptypes = [rptype_map[k] for k in ptypes]
        for ptype in ptypes:
            my_ptype = ptype_map[ptype]
            particle_types.append(my_ptype)
            g = f[ptype]
            for field in gadget_fields[my_ptype]:
                if field in g:
                    if field == "ParticleIDs":
                        fields[my_ptype, "particle_index"] = g[field][:]
                    else:
                        fd = gadget_field_map[field]
                        units = gadget_field_units[field]
                        fields[my_ptype, fd] = YTArray(g[field], units, dtype='float64').in_base("galactic")
            if "Masses" not in g:
                n_ptype = g["ParticleIDs"].size
                units = gadget_field_units["Masses"]
                n_type = int(ptype[-1])
                fields[my_ptype, "particle_mass"] = YTArray([f["Header"].attrs["MassTable"][n_type]]*n_ptype,
                                                            units).in_base("galactic")
        box_size = f["/Header"].attrs["BoxSize"]
        f.close()
        return cls(particle_types, fields, box_size=box_size)

    @classmethod
    def from_gamer_output(cls, filename, ptype="dm"):
        fields = OrderedDict()
        particle_types = [ptype]
        f = h5py.File(filename, "r")
        g = f["Particle"]
        lunit = f["Info"]["InputPara"]["Unit_L"].value
        munit = f["Info"]["InputPara"]["Unit_M"].value
        vunit = lunit/f["Info"]["InputPara"]["Unit_T"]
        fields[ptype, "particle_mass"] = YTArray(
            g["ParMass"][:]*munit, "g").in_base("galactic")
        fields[ptype, "particle_position"] = YTArray(
            [g[f"ParPos{ax}"][:]*lunit for ax in "XYZ"],
            "cm").in_base("galactic")
        fields[ptype, "particle_velocity"] = YTArray(
            [g[f"ParVel{ax}"][:]*vunit for ax in "XYZ"],
            "cm/s").in_base("galactic")
        f.close()
        return cls(particle_types, fields)

    def _update_num_particles(self):
        self.num_particles = {}
        for ptype in self.particle_types:
            self.num_particles[ptype] = self.fields[ptype, "particle_mass"].size

    def _update_field_names(self):
        self.field_names = defaultdict(list)
        for field in self.fields:
            self.field_names[field[0]].append(field[1])

    def drop_ptypes(self, ptypes):
        """
        Drop all particles with a type in *ptypes*.
        """
        ptypes = ensure_list(ptypes)
        for ptype in ptypes:
            self.particle_types.remove(ptype)
            names = list(self.fields.keys())
            for name in names:
                if name[0] in ptypes:
                   self.fields.pop(name) 
        self._update_num_particles()
        self._update_field_names()
        
    def make_radial_cut(self, r_max, center=None, ptypes=None):
        """
        Make a radial cut on particles. All particles outside
        a certain radius will be removed.

        Parameters
        ----------
        r_max : float
            The maximum radius of the particles in kpc.
        center : array-like, optional
            The center coordinate of the system of particles to define
            the radius from, in units of kpc. Default: [0.0, 0.0, 0.0]
        ptypes : list of strings, optional
            The particle types to perform the radial cut on. If
            not set, all will be exported.
        """
        rm2 = r_max*r_max
        if center is None:
            center = np.array([0.0]*3)
        if ptypes is None:
            ptypes = self.particle_types
        ptypes = ensure_list(ptypes)

        for pt in ptypes:
            cidx = ((self[pt, "particle_position"].d-center)**2).sum(axis=1) <= rm2
            for field in self.field_names[pt]:
                self.fields[pt, field] = self.fields[pt, field][cidx]
        self._update_num_particles()

    def add_black_hole(self, bh_mass, pos=None, vel=None,
                       use_pot_min=False):
        r"""
        Add a black hole particle to the set of cluster
        particles.

        Parameters
        ----------
        bh_mass : float
            The mass of the black hole particle in solar masses.
        pos : array-like, optional
            The position of the particle, assumed to be in units of
            kpc if units are not given. If use_pot_min=True this
            argument is ignored. Default: None, in which case the 
            particle position is [0.0, 0.0, 0.0] kpc. 
        vel : array-like, optional
            The velocity of the particle, assumed to be in units of
            kpc/Myr if units are not given. If use_pot_min=True this
            argument is ignored. Default: None, in which case the 
            particle velocity is [0.0, 0.0, 0.0] kpc/Myr. 
        use_pot_min : boolean, optional 
            If True, use the dark matter particle with the minimum
            value of the gravitational potential to determine the 
            position and velocity of the black hole particle. Default:
            False
        """
        mass = YTQuantity(bh_mass, "Msun")
        self.fields["black_hole", "particle_mass"] = YTArray([bh_mass], "Msun")
        if use_pot_min:
            idx = np.argmin(self.fields["dm", "potential_energy"])
            pos = YTArray(self.fields["dm", "particle_position"][idx]).reshape(1,3)
            vel = YTArray(self.fields["dm", "particle_velocity"][idx]).reshape(1,3)
        else:
            if pos is None:
                pos = YTArray(np.zeros((1,3)), "kpc")
            if vel is None:
                vel = YTArray(np.zeros((1,3)), "kpc/Myr")
            pos = ensure_ytarray(pos, "kpc")
            vel = ensure_ytarray(vel, "kpc/Myr")
        if "black_hole" not in self.particle_types:
            self.particle_types.append("black_hole")
            self.fields["black_hole", "particle_position"] = pos
            self.fields["black_hole", "particle_velocity"] = vel
            self.fields["black_hole", "particle_mass"] = mass
        else:
            uappend = lambda x, y: YTArray(np.append(x, y, axis=0).v, x.units)
            self.fields["black_hole", "particle_position"] = uappend(
                self.fields["black_hole", "particle_position"], pos)
            self.fields["black_hole", "particle_velocity"] = uappend(
                self.fields["black_hole", "particle_velocity"], vel)
            self.fields["black_hole", "particle_mass"] = uappend(
                self.fields["black_hole", "particle_mass"], mass)
        self._update_num_particles()

    def write_particles_to_h5(self, output_filename, in_cgs=False, 
                              overwrite=False):
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
        with h5py.File(output_filename, "w") as f:
            for ptype in self.particle_types:
                f.create_group(ptype)
        for field in self.fields:
            if field[1] == "particle_index":
                with h5py.File(output_filename, "r+") as f:
                    g = f[field[0]]
                    g.create_dataset("particle_index", data=self.fields[field])
            else:
                if in_cgs:
                    fd = self.fields[field].in_cgs()
                else:
                    fd = self.fields[field]
                    fd.write_hdf5(output_filename, dataset_name=field[1],
                                  group_name=field[0])

    def write_gamer_input(self, output_filename, overwrite=True):
        """
        Write the particles to an HDF5 file to be read in by the GAMER
        code.

        Parameters
        ----------
        output_filename : string
            The file to write the particles to.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError("Cannot create %s. It exists and overwrite=False." % output_filename)
        ptypes = ["dm"]
        if "star" in self.particle_types:
            ptypes.append("star")
        nparts = [self.num_particles[ptype] for ptype in ptypes]
        f = h5py.File(output_filename, "w")
        for field in self.field_names["dm"]:
            fd = uconcatenate([self.fields[ptype, field] for ptype in ptypes], axis=0)
            if hasattr(fd, "units"):
                print(field, fd.units)
                fd.convert_to_cgs()
            f.create_dataset(field, data=np.asarray(fd))
        fd = np.concatenate([(i+1)*np.ones(nparts[i]) for i, ptype in enumerate(ptypes)])
        f.create_dataset("particle_type", data=fd)
        f.flush()
        f.close()

    def __add__(self, other):
        fields = self.fields.copy()
        for field in other.fields:
            if field in fields:
                fields[field] = uconcatenate([self[field], other[field]])
            else:
                fields[field] = other[field]
        particle_types = list(set(self.particle_types + other.particle_types))
        return ClusterParticles(particle_types, fields)

    @property
    def num_passive_scalars(self):
        return len(self.passive_scalars)

    def add_offsets(self, r_ctr, v_ctr, ptypes=None):
        """
        Add offsets in position and velocity to the cluster particles,
        which can be added to one or more particle types.

        Parameters
        ----------
        r_ctr : array-like
            A 3-element list, NumPy array, or YTArray of the coordinates
            of the new center of the particle distribution.
        v_ctr : array-like
            A 3-element list, NumPy array, or YTArray of the coordinates
            of the new bulk velocity of the particle distribution.
        ptypes : string or list of strings, optional
            A single string or list of strings indicating the particle
            type(s) to be offset. Default: None, meaning all of the 
            particle types will be offset.
        """
        if ptypes is None:
            ptypes = self.particle_types
        ptypes = ensure_list(ptypes)
        r_ctr = ensure_ytarray(r_ctr, "kpc")
        v_ctr = ensure_ytarray(v_ctr, "kpc/Myr")
        for ptype in ptypes:
            self.fields[ptype, "particle_position"] += r_ctr
            self.fields[ptype, "particle_velocity"] += v_ctr

    def _clip_to_box(self, ptype, box_size):
        pos = self.fields[ptype, "particle_position"]
        return ~np.logical_or((pos < 0.0).any(axis=1),
                              (pos > box_size).any(axis=1))

    def _write_gadget_fields(self, ptype, h5_group, idxs, dtype):
        for field in gadget_fields[ptype]:
            if field == "ParticleIDs":
                continue
            if field == "PassiveScalars" and ptype == "gas":
                if self.num_passive_scalars > 0:
                    data = np.stack(
                        [self[ptype, s].d for s in self.passive_scalars], 
                        axis=-1)
                    h5_group.create_dataset("PassiveScalars", data=data)
            else:
                my_field = gadget_field_map[field]
                if (ptype, my_field) in self.fields:
                    units = gadget_field_units[field]
                    fd = self.fields[ptype, my_field]
                    data = fd[idxs].to(units).d.astype(dtype)
                    h5_group.create_dataset(field, data=data)

    def write_to_gadget_file(self, ic_filename, box_size, 
                             dtype='float32', overwrite=False):
        """
        Write the particles to a file in the HDF5 Gadget format
        which can be used as initial conditions for a simulation.

        Parameters
        ----------
        ic_filename : string
            The name of the file to write to.
        box_size : float
            The width of the cubical box that the initial condition
            will be within in units of kpc. 
        dtype : string, optional
            The datatype of the fields to write, either 'float32' or
            'float64'. Default: 'float32'
        overwrite : boolean, optional
            Whether or not to overwrite an existing file. Default: False
        """
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
            if ptype in ["star", "dm", "black_hole"]:
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
                                                 num_particles.get("black_hole", 0)],
                                                dtype='uint32')
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

    def set_field(self, ptype, name, value, units=None, add=False,
                  passive_scalar=False):
        """
        Add or update a particle field using a YTArray.
        The array will be checked to make sure that it
        has the appropriate size.

        Parameters
        ----------
        ptype : string
            The particle type of the field to add or update.
        name : string
            The name of the field to add or update.
        value : YTArray
            The particle field itself--an array with the same 
            shape as the number of particles.
        units : string, optional
            The units to convert the field to. Default: None,
            indicating the units will be preserved.
        add : boolean, optional
            If True and the field already exists, the values
            in the array will be added to the already existing
            field array.
        """
        if not isinstance(value, YTArray):
            value = YTArray(value, "dimensionless")
        num_particles = self.num_particles[ptype]
        exists = (ptype, name) in self.fields
        if value.shape[0] == num_particles:
            if exists:
                if add:
                    self.fields[ptype, name] += value
                else:
                    mylog.warning(f"Overwriting field ({ptype}, {name}).")
                    self.fields[ptype, name] = value
            else:
                if add:
                    raise RuntimeError(f"Field ({ptype}, {name}) does not "
                                       f"exist and add=True!")
                else:
                    self.fields[ptype, name] = value
                if passive_scalar and ptype == "gas":
                    self.passive_scalars.append(name)
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

    def to_yt_dataset(self, box_size, ptypes=None):
        """
        Create an in-memory yt dataset for the particles. 

        Parameters
        ----------
        box_size : float
            The width of the domain on a side, in kpc.
        ptypes : list of strings, optional
            The particle types to export to the dataset. If
            not set, all will be exported.
        """
        data = self.fields.copy()
        if ptypes is None:
            ptypes = self.particle_types
        ptypes = ensure_list(ptypes)
        for ptype in ptypes:
            pos = data.pop((ptype, "particle_position"))
            vel = data.pop((ptype, "particle_velocity"))
            for i, ax in enumerate("xyz"):
                data[ptype, "particle_position_%s" % ax] = pos[:,i]
                data[ptype, "particle_velocity_%s" % ax] = vel[:,i]
        return load_particles(data, length_unit="kpc", bbox=[[0.0, box_size]]*3,
                              mass_unit="Msun", time_unit="Myr")


def _sample_clusters(particles, hses, center, velocity,
                     radii=None, resample=False, 
                     passive_scalars=None):
    num_halos = len(hses)
    center = [ensure_ytarray(c, "kpc") for c in center]
    velocity = [ensure_ytarray(v, "kpc/Myr") for v in velocity]
    r = np.zeros((num_halos, particles.num_particles["gas"]))
    for i, c in enumerate(center):
        r[i,:] = ((particles["gas", "particle_position"]-c)**2).sum(axis=1).d
    np.sqrt(r, r)
    if radii is None:
        idxs = slice(None, None, None)
    else:
        radii = np.array(radii)
        idxs = np.any(r <= radii[:,np.newaxis], axis=0)
    d = np.zeros((num_halos, particles.num_particles["gas"]))
    e = np.zeros((num_halos, particles.num_particles["gas"]))
    m = np.zeros((num_halos, 3, particles.num_particles["gas"]))
    num_scalars = 0
    if passive_scalars is not None:
        num_scalars = len(passive_scalars)
        s = np.zeros((num_halos, num_scalars, particles.num_particles["gas"]))
    for i in range(num_halos):
        hse = hses[i]
        get_density = InterpolatedUnivariateSpline(hse["radius"], hse["density"])
        d[i,:] = get_density(r[i,:])
        e_arr = 1.5*hse["pressure"]/hse["density"]
        get_energy = InterpolatedUnivariateSpline(hse["radius"], e_arr)
        e[i,:] = get_energy(r[i,:])*d[i,:]
        m[i,:,:] = velocity[i].d[:,np.newaxis]*d[i,:]
        if num_scalars > 0:
            for j, name in enumerate(passive_scalars):
                get_scalar = InterpolatedUnivariateSpline(hse["radius"], hse[name])
                s[i,j,:] = get_scalar(r[i,:])*d[i,:]
    dens = d.sum(axis=0)
    eint = e.sum(axis=0)/dens
    mom  = m.sum(axis=0)/dens
    if num_scalars > 0:
        ps = s.sum(axis=0)/dens
    if resample:
        vol = particles["gas", "particle_mass"]/particles["gas", "density"]
        particles["gas", "particle_mass"][idxs] = dens[idxs]*vol[idxs]
    particles["gas", "density"][idxs] = dens[idxs]
    particles["gas", "thermal_energy"][idxs] = eint[idxs]
    particles["gas", "particle_velocity"][idxs] = mom.T[idxs]
    if num_scalars > 0:
        for j, name in enumerate(passive_scalars):
            particles["gas", name][idxs] = ps[j,idxs]
    return particles


def combine_two_clusters(particles1, particles2, hse1, hse2,
                         center1, center2, velocity1, velocity2):
    center1 = ensure_ytarray(center1, "kpc")
    center2 = ensure_ytarray(center2, "kpc")
    velocity1 = ensure_ytarray(velocity1, "kpc/Myr")
    velocity2 = ensure_ytarray(velocity2, "kpc/Myr")
    particles1.add_offsets(center1, [0.0]*3, ptypes=["gas"])
    particles2.add_offsets(center2, [0.0]*3, ptypes=["gas"])
    ptypes = ["dm"]
    if "star" in particles1.particle_types:
        ptypes.append("star")
    if "black_hole" in particles1.particle_types:
        ptypes.append("black_hole")
    particles1.add_offsets(center1, velocity1, ptypes=ptypes)
    particles2.add_offsets(center2, velocity2, ptypes=ptypes)
    particles = particles1+particles2
    particles = _sample_clusters(particles, [hse1, hse2], 
                                 [center1, center2], [velocity1, velocity2])
    return particles


def combine_three_clusters(particles1, particles2, particles3,
                           hse1, hse2, hse3, center1, center2,
                           center3, velocity1, velocity2,
                           velocity3):
    center1 = ensure_ytarray(center1, "kpc")
    center2 = ensure_ytarray(center2, "kpc")
    center3 = ensure_ytarray(center3, "kpc")
    velocity1 = ensure_ytarray(velocity1, "kpc/Myr")
    velocity2 = ensure_ytarray(velocity2, "kpc/Myr")
    velocity3 = ensure_ytarray(velocity3, "kpc/Myr")
    particles1.add_offsets(center1, [0.0]*3, ptypes=["gas"])
    particles2.add_offsets(center2, [0.0]*3, ptypes=["gas"])
    particles3.add_offsets(center3, [0.0]*3, ptypes=["gas"])
    ptypes = ["dm"]
    if "star" in particles1.particle_types:
        ptypes.append("star")
    if "black_hole" in particles1.particle_types:
        ptypes.append("black_hole")
    particles1.add_offsets(center1, velocity1, ptypes=ptypes)
    particles2.add_offsets(center2, velocity2, ptypes=ptypes)
    particles3.add_offsets(center3, velocity3, ptypes=ptypes)
    particles = particles1+particles2+particles3
    particles = _sample_clusters(particles, [hse1, hse2, hse3],
                                 [center1, center2, center3],
                                 [velocity1, velocity2, velocity3])
    return particles


def resample_one_cluster(particles, hse, center, velocity):
    center = ensure_ytarray(center, "kpc")
    velocity = ensure_ytarray(velocity, "kpc/Myr")
    r = ((particles["gas", "particle_position"]-center)**2).sum(axis=1).d
    np.sqrt(r, r)
    get_density = InterpolatedUnivariateSpline(hse["radius"], hse["density"])
    dens = get_density(r)
    e_arr = 1.5 * hse["pressure"] / hse["density"]
    get_energy = InterpolatedUnivariateSpline(hse["radius"], e_arr)
    particles["gas", "thermal_energy"] = YTArray(get_energy(r), "kpc**2/Myr**2")
    vol = particles["gas", "particle_mass"] / particles["gas", "density"]
    particles["gas", "particle_mass"] = YTArray(dens*vol.d, "Msun")
    particles["gas", "particle_velocity"][:,:] = velocity
    particles["gas", "density"] = YTArray(dens, "Msun/kpc**3")
    return particles


def resample_two_clusters(particles, hse1, hse2, center1, center2,
                          velocity1, velocity2, radii,
                          passive_scalars=None):
    particles = _sample_clusters(particles, [hse1, hse2],
                                 [center1, center2],
                                 [velocity1, velocity2],
                                 radii=radii, resample=True,
                                 passive_scalars=passive_scalars)
    return particles


def resample_three_clusters(particles, hse1, hse2, hse3, center1,
                            center2, center3, velocity1, velocity2,
                            velocity3, radii, passive_scalars=None):
    particles = _sample_clusters(particles, [hse1, hse2, hse3],
                                 [center1, center2, center3],
                                 [velocity1, velocity2, velocity3],
                                 radii=radii, resample=True,
                                 passive_scalars=passive_scalars)
    return particles
