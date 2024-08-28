"""Initial conditions for use in simulation codes.

Notes
-----

In effect, the :py:class:`ClusterICs` class allows for the combination of individual :py:class:`model.ClusterModel` instances
into a single, interacting system for use in external codes.
"""
import os
from numbers import Number

import numpy as np
from ruamel.yaml import YAML

from cluster_generator.model import ClusterModel
from cluster_generator.particles import (
    ClusterParticles,
    combine_three_clusters,
    combine_two_clusters,
    resample_one_cluster,
    resample_three_clusters,
    resample_two_clusters,
)
from cluster_generator.utils import ensure_list, ensure_ytarray, parse_prng


def compute_centers_for_binary(center, d, b, a=0.0):
    r"""Given a common center and distance parameters, calculate the central positions of
    two clusters.

    First, the separation along the x-direction is determined
    by:

    .. math::

        x = \sqrt{d^2 - b^2 - a^2}

    where :math:`d` is the distance between the two clusters, :math:`b` is the
    impact parameter in the y-direction, and :math:`a` is the impact
    parameter in the z-direction. So the resulting centers are
    calculated as:

    .. code-block:: python

        center1 = [center-0.5*sep_x, center-0.5*b, center-0.5*a]
        center2 = [center+0.5*sep_x, center+0.5*b, center+0.5*a]

    Parameters
    ----------
    center : array-like
        The center from which the distance parameters for
        the two clusters will be calculated. This should be a ``(3,)`` array of
        coordinates.

    d : float
        The distance between the two clusters, in kpc.
    b : float
        The impact parameter in the y-direction, in kpc.
    a : float, optional
        The impact parameter in the z-direction, in kpc.
        Default: 0.0

    Examples
    --------

    If we have 2 clusters separated by 5000 kpc and with an impact parameter of 3000 kpc, then we expect that the
    :math:`x` separation should be 4000 kpc. Thus,

    >>> compute_centers_for_binary([0,0,0],5000,3000)
    (array([-2000., -1500.,     0.]), array([2000., 1500.,    0.]))
    """
    d = np.sqrt(d * d - b * b - a * a)
    diff = np.array([d, b, a])
    center1 = center - 0.5 * diff
    center2 = center + 0.5 * diff
    return center1, center2


class ClusterICs:
    """Class representing a complete set of initial conditions for a simulation."""

    def __init__(
        self,
        basename,
        num_halos,
        profiles,
        center,
        velocity,
        num_particles=None,
        mag_file=None,
        particle_files=None,
        r_max=20000.0,
        r_max_tracer=None,
    ):
        """Initialize a :py:class:`ClusterICs` instance.

        Parameters
        ----------
        basename: str
            The base name for this initial conditions file.
        num_halos: int
            The number of halos in this initial conditions instance.
        profiles: Collection[str]
            The paths to each of the constituent clusters.
        center: array-like
            The center of each of the clusters. Should be ``(N,3)``.
        velocity: array-like
            The velocity of each of the clusters. Should be ``(N,3)``.
        num_particles: dict of str: int, optional
            The number of particles associated with each of the particle species.
        mag_file: str
            The file containing the magnetic field prescription.
        particle_files: list[str]
            The files containing the particle data for each of the IC components.
        r_max: array-like or float, optional
            The maximal radius for each of the constituent clusters.
        r_max_tracer
        """
        self.basename = basename
        """ str: The name of this IC system."""
        self.num_halos = num_halos
        """ int: The number of halos present in the IC system."""
        self.profiles = ensure_list(profiles)
        """ list of Path: The paths of the constituent profiles."""
        self.center = ensure_ytarray(center, "kpc")
        """ list of unyt_array: The centers of the constituent profiles.

        This should be a ``(N,3)`` list where ``N`` is the number of profiles included.
        """
        self.velocity = ensure_ytarray(velocity, "kpc/Myr")
        """ list of unyt_array: The velocities of the constituent profiles.

        This should be a ``(N,3)`` list where ``N`` is the number of profiles included.
        """
        if self.num_halos == 1:
            self.center = self.center.reshape(1, 3)
            self.velocity = self.velocity.reshape(1, 3)
        self.mag_file = mag_file
        """ Path: The file containing the magnetic field information for the ICs."""
        if isinstance(r_max, Number):
            r_max = [r_max] * num_halos
        self.r_max = np.array(r_max)
        """:py:class:`np.ndarray`: The maximal radii for each of the constituent
        clusters."""
        if r_max_tracer is None:
            r_max_tracer = r_max
        if isinstance(r_max_tracer, Number):
            r_max_tracer = [r_max_tracer] * num_halos
        self.r_max_tracer = np.array(r_max_tracer)
        """:py:class:`np.ndarray`: The maximal (tracer) radii for each of the
        constituent clusters."""
        if num_particles is None:
            self.tot_np = {"dm": 0, "gas": 0, "star": 0, "tracer": 0}
        else:
            self.tot_np = num_particles
        self._determine_num_particles()
        self.particle_files = [None] * 3
        """ list: The paths of the different particle files for each cluster.
        """
        if particle_files is not None:
            self.particle_files[:num_halos] = particle_files[:]

    def __repr__(self):
        return f"<ClusterICs: {self.num_halos} models>"

    def __str__(self):
        return self.__repr__()

    def _determine_num_particles(self):
        """Determines the number of particles of each type to be attributed to each
        cluster.
        """
        from collections import defaultdict

        dm_masses = []
        gas_masses = []
        star_masses = []
        tracer_masses = []
        for i, pf in enumerate(self.profiles):
            p = ClusterModel.from_h5_file(pf)
            idxs = p["radius"] < self.r_max[i]
            dm_masses.append(p["dark_matter_mass"][idxs][-1].value)
            if "gas_mass" in p:
                gmass = p["gas_mass"][idxs][-1].value
            else:
                gmass = 0.0
            gas_masses.append(gmass)
            if "stellar_mass" in p:
                smass = p["stellar_mass"][idxs][-1].value
            else:
                smass = 0.0
            star_masses.append(smass)
            if self.tot_np.get("tracer", 0) > 0:
                idxst = p["radius"] < self.r_max_tracer[i]
                tmass = p["gas_mass"][idxst][-1].value
            else:
                tmass = 0.0
            tracer_masses.append(tmass)
        tot_dm_mass = np.sum(dm_masses)
        tot_gas_mass = np.sum(gas_masses)
        tot_star_mass = np.sum(star_masses)
        tot_tracer_mass = np.sum(tracer_masses)
        self.num_particles = defaultdict(list)
        """ dict: The number of particles of each species in each cluster.

        This is calculated at instantiation from the provided number of particles and the cluster masses.
        """
        for i in range(self.num_halos):
            if self.tot_np.get("dm", 0) > 0:
                ndp = np.rint(self.tot_np["dm"] * dm_masses[i] / tot_dm_mass).astype(
                    "int"
                )
            else:
                ndp = 0
            self.num_particles["dm"].append(ndp)
            if self.tot_np.get("gas", 0) > 0:
                ngp = np.rint(self.tot_np["gas"] * gas_masses[i] / tot_gas_mass).astype(
                    "int"
                )
            else:
                ngp = 0
            self.num_particles["gas"].append(ngp)
            if self.tot_np.get("star", 0) > 0:
                nsp = np.rint(
                    self.tot_np["star"] * star_masses[i] / tot_star_mass
                ).astype("int")
            else:
                nsp = 0
            self.num_particles["star"].append(nsp)
            if self.tot_np.get("tracer", 0) > 0:
                ntp = np.rint(
                    self.tot_np["tracer"] * tracer_masses[i] / tot_tracer_mass
                ).astype("int")
            else:
                ntp = 0
            self.num_particles["tracer"].append(ntp)

    def _generate_particles(self, regenerate_particles=False, prng=None):
        """Generate particles for the initial conditions."""
        prng = parse_prng(prng)
        parts = []
        for i, pf in enumerate(self.profiles):
            if regenerate_particles or self.particle_files[i] is None:
                m = ClusterModel.from_h5_file(pf)
                p = m.generate_dm_particles(
                    self.num_particles["dm"][i], r_max=self.r_max[i], prng=prng
                )
                if self.num_particles["star"][i] > 0:
                    sp = m.generate_star_particles(
                        self.num_particles["star"][i], r_max=self.r_max[i], prng=prng
                    )
                    p = p + sp
                if self.num_particles["gas"][i] > 0:
                    gp = m.generate_gas_particles(
                        self.num_particles["gas"][i], r_max=self.r_max[i], prng=prng
                    )
                    p = p + gp
                if self.num_particles["tracer"][i] > 0:
                    tp = m.generate_tracer_particles(
                        self.num_particles["tracer"][i],
                        r_max=self.r_max_tracer[i],
                        prng=prng,
                    )
                    p = p + tp
                parts.append(p)
                outfile = f"{self.basename}_{i}_particles.h5"
                p.write_particles(outfile, overwrite=True)
                self.particle_files[i] = outfile
            else:
                p = ClusterParticles.from_file(self.particle_files[i])
                parts.append(p)
        return parts

    def to_file(self, filename, overwrite=False):
        r"""Write the initial conditions information to a file.

        Parameters
        ----------
        filename : string
            The file to write the initial conditions information to.
        overwrite : boolean, optional
            If True, overwrite a file with the same name. Default: False

        Notes
        -----
        The file representation for :py:class:`ClusterICs` is a ``.yaml`` file effectively redirecting the information
        to other files storing the particles and the models.
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        from ruamel.yaml.comments import CommentedMap

        out = CommentedMap()
        out["basename"] = self.basename
        out.yaml_add_eol_comment("base name for ICs", key="basename")
        out["num_halos"] = self.num_halos
        out.yaml_add_eol_comment("number of halos", key="num_halos")
        out["profile1"] = self.profiles[0]
        out.yaml_add_eol_comment("profile for cluster 1", key="profile1")
        out["center1"] = self.center[0].tolist()
        out.yaml_add_eol_comment("center for cluster 1", key="center1")
        out["velocity1"] = self.velocity[0].tolist()
        out.yaml_add_eol_comment("velocity for cluster 1", key="velocity1")
        if self.particle_files[0] is not None:
            out["particle_file1"] = self.particle_files[0]
            out.yaml_add_eol_comment(
                "particle file for cluster 1", key="particle_file1"
            )
        if self.num_halos > 1:
            out["profile2"] = self.profiles[1]
            out.yaml_add_eol_comment("profile for cluster 2", key="profile2")
            out["center2"] = self.center[1].tolist()
            out.yaml_add_eol_comment("center for cluster 2", key="center2")
            out["velocity2"] = self.velocity[1].tolist()
            out.yaml_add_eol_comment("velocity for cluster 2", key="velocity2")
            if self.particle_files[1] is not None:
                out["particle_file2"] = self.particle_files[1]
                out.yaml_add_eol_comment(
                    "particle file for cluster 2", key="particle_file2"
                )
        if self.num_halos == 3:
            out["profile3"] = self.profiles[2]
            out.yaml_add_eol_comment("profile for cluster 3", key="profile3")
            out["center3"] = self.center[2].tolist()
            out.yaml_add_eol_comment("center for cluster 3", key="center3")
            out["velocity3"] = self.velocity[2].tolist()
            out.yaml_add_eol_comment("velocity for cluster 3", key="velocity3")
            if self.particle_files[2] is not None:
                out["particle_file3"] = self.particle_files[2]
                out.yaml_add_eol_comment(
                    "particle file for cluster 3", key="particle_file3"
                )
        if self.tot_np.get("dm", 0) > 0:
            out["num_dm_particles"] = self.tot_np["dm"]
            out.yaml_add_eol_comment("number of DM particles", key="num_dm_particles")
        if self.tot_np.get("gas", 0) > 0:
            out["num_gas_particles"] = self.tot_np["gas"]
            out.yaml_add_eol_comment("number of gas particles", key="num_gas_particles")
        if self.tot_np.get("star", 0) > 0:
            out["num_star_particles"] = self.tot_np["star"]
            out.yaml_add_eol_comment(
                "number of star particles", key="num_star_particles"
            )
        if self.tot_np.get("tracer", 0) > 0:
            out["num_tracer_particles"] = self.tot_np["tracer"]
            out.yaml_add_eol_comment(
                "number of tracer particles", key="num_tracer_particles"
            )
        if self.mag_file is not None:
            out["mag_file"] = self.mag_file
            out.yaml_add_eol_comment("3D magnetic field file", key="mag_file")
        out["r_max"] = self.r_max.tolist()
        out.yaml_add_eol_comment("Maximum radii of particles", key="r_max")
        if self.tot_np.get("tracer", 0) > 0:
            out["r_max_tracer"] = self.r_max_tracer.tolist()
            out.yaml_add_eol_comment("Maximum radii of tracer particles", key="r_max")
        yaml = YAML()
        with open(filename, "w") as f:
            yaml.dump(out, f)

    @classmethod
    def from_file(cls, filename):
        r"""Read a :py:class:`ClusterICs` instance from a ``.yaml`` file.

        Parameters
        ----------
        filename: str
            The path to the ``.yaml`` file.

        Returns
        -------
        ClusterICs
            The resulting initial conditions object.
        """
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(filename, "r") as f:
            params = yaml.load(f)
        basename = params["basename"]
        num_halos = params["num_halos"]
        profiles = [params[f"profile{i}"] for i in range(1, num_halos + 1)]
        center = [np.array(params[f"center{i}"]) for i in range(1, num_halos + 1)]
        velocity = [np.array(params[f"velocity{i}"]) for i in range(1, num_halos + 1)]
        num_particles = {
            k: params.get(f"num_{k}_particles", 0) for k in ["gas", "dm", "star"]
        }
        mag_file = params.get("mag_file", None)
        particle_files = [
            params.get(f"particle_file{i}", None) for i in range(1, num_halos + 1)
        ]
        r_max = params.get("r_max", 20000.0)
        r_max_tracer = params.get("r_max_tracer", r_max)
        return cls(
            basename,
            num_halos,
            profiles,
            center,
            velocity,
            num_particles=num_particles,
            mag_file=mag_file,
            particle_files=particle_files,
            r_max=r_max,
            r_max_tracer=r_max_tracer,
        )

    def setup_particle_ics(self, regenerate_particles=False, prng=None):
        r"""From a set of cluster models and their relative positions and velocities, set
        up initial conditions for use with SPH codes.

        Parameters
        ----------
        regenerate_particles: bool
            If ``True``, then existing particle files are overwritten and resampled.
        prng: RandomState or int
            The pseudo-random number generator seed or instance.

        Returns
        -------
        ClusterParticles
            The combined particle dataset ready for use in SPH codes.

        Notes
        -----

        This routine will either generate a single cluster or will combine
        two or three clusters together. If more than one cluster is
        generated, the gas particles will have their densities set by
        adding the densities from the overlap of the two particles
        together, and will have their thermal energies and velocities
        set by mass-weighting them from the two profiles.
        """
        profiles = [ClusterModel.from_h5_file(hf) for hf in self.profiles]
        parts = self._generate_particles(
            regenerate_particles=regenerate_particles, prng=prng
        )
        if self.num_halos == 1:
            all_parts = parts[0]
            all_parts.add_offsets(self.center[0], self.velocity[0])
        elif self.num_halos == 2:
            all_parts = combine_two_clusters(
                parts[0],
                parts[1],
                profiles[0],
                profiles[1],
                self.center[0],
                self.center[1],
                self.velocity[0],
                self.velocity[1],
            )
        else:
            all_parts = combine_three_clusters(
                parts[0],
                parts[1],
                parts[2],
                profiles[0],
                profiles[1],
                profiles[2],
                self.center[0],
                self.center[1],
                self.center[2],
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
            )
        return all_parts

    def resample_particle_ics(self, parts, passive_scalars=None):
        r"""Given a Gadget-HDF5-like initial conditions file which has been output from
        some type of relaxation process (such as making a glass or using MESHRELAX in
        the case of Arepo), resample the density, thermal energy, and velocity fields
        onto the gas particles/cells from the initial hydrostatic profiles.

        Parameters
        ----------
        particles: ClusterParticles
            The particle dataset to resample onto.
        passive_scalars: list of str
            Any passive scalar fields to pull from the models and map onto the particles.
        """
        profiles = [ClusterModel.from_h5_file(hf) for hf in self.profiles]
        if self.num_halos == 1:
            new_parts = resample_one_cluster(
                parts, profiles[0], self.center[0], self.velocity[0]
            )
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(
                parts,
                profiles[0],
                profiles[1],
                self.center[0],
                self.center[1],
                self.velocity[0],
                self.velocity[1],
                self.r_max,
                passive_scalars=passive_scalars,
            )
        else:
            new_parts = resample_three_clusters(
                parts,
                profiles[0],
                profiles[1],
                profiles[2],
                self.center[0],
                self.center[1],
                self.center[2],
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
                self.r_max,
                passive_scalars=passive_scalars,
            )
        return new_parts

    def create_dataset(self, domain_dimensions, box_size, left_edge=None, **kwargs):
        """
        Create an in-memory, uniformly gridded dataset in 3D using yt by
        placing the clusters into a box. When adding multiple clusters,
        per-volume quantities from each cluster such as density and
        pressure are added, whereas per-mass quantities such as temperature
        and velocity are mass-weighted.

        Parameters
        ----------
        domain_dimensions : 3-tuple of ints
            The number of cells on a side for the domain.
        box_size : float
            The size of the box in kpc.
        left_edge : array_like, optional
            The minimum coordinate of the box in all three dimensions,
            in kpc. Default: None, which means the left edge will
            be [0, 0, 0].
        """
        from scipy.interpolate import InterpolatedUnivariateSpline
        from yt.loaders import load_uniform_grid

        if left_edge is None:
            left_edge = np.zeros(3)
        left_edge = np.array(left_edge)
        bbox = [
            [left_edge[0], left_edge[0] + box_size],
            [left_edge[1], left_edge[1] + box_size],
            [left_edge[2], left_edge[2] + box_size],
        ]
        x, y, z = np.mgrid[
            bbox[0][0] : bbox[0][1] : domain_dimensions[0] * 1j,
            bbox[1][0] : bbox[1][1] : domain_dimensions[1] * 1j,
            bbox[2][0] : bbox[2][1] : domain_dimensions[2] * 1j,
        ]
        fields1 = [
            "density",
            "pressure",
            "dark_matter_density" "stellar_density",
            "gravitational_potential",
        ]
        fields2 = ["temperature"]
        fields3 = ["velocity_x", "velocity_y", "velocity_z"]
        units = {
            "density": "Msun/kpc**3",
            "pressure": "Msun/kpc/Myr**2",
            "dark_matter_density": "Msun/kpc**3",
            "stellar_density": "Msun/kpc**3",
            "temperature": "K",
            "gravitational_potential": "kpc**2/Myr**2",
            "velocity_x": "kpc/Myr",
            "velocity_y": "kpc/Myr",
            "velocity_z": "kpc/Myr",
            "magnetic_field_strength": "G",
        }
        fields = fields1 + fields2
        data = {}
        for i, profile in enumerate(self.profiles):
            p = ClusterModel.from_h5_file(profile)
            xx = x - self.center.d[i][0]
            yy = y - self.center.d[i][1]
            zz = z - self.center.d[i][2]
            rr = np.sqrt(xx * xx + yy * yy + zz * zz)
            fd = InterpolatedUnivariateSpline(p["radius"].d, p["density"].d)
            for field in fields:
                if field not in p:
                    continue
                if field not in data:
                    data[field] = (np.zeros(domain_dimensions), units[field])
                f = InterpolatedUnivariateSpline(p["radius"].d, p[field].d)
                if field in fields1:
                    data[field][0] += f(rr)
                elif field in fields2:
                    data[field][0] += f(rr) * fd(rr)
            for field in fields3:
                data[field][0] += self.velocity.d[i][0] * fd(rr)
        if "density" in data:
            for field in fields2 + fields3:
                data[field][0] /= data["density"][0]
        return load_uniform_grid(
            data,
            domain_dimensions,
            length_unit="kpc",
            bbox=bbox,
            mass_unit="Msun",
            time_unit="Myr",
            **kwargs,
        )
