"""
Virialization tools for the ``cluster-generator`` system.
"""
import numpy as np
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf
from cluster_generator.utils import \
    quad, generate_particle_radii, mylog
from cluster_generator.particles import \
    ClusterParticles
from cluster_generator.cython_utils import generate_velocities, generate_lma_velocities
from collections import OrderedDict
from unyt import unyt_array
from scipy.optimize import fsolve

class VirialEquilibrium:
    r"""
    The ``VirialEquilibrium`` class is used to generate the necessary distribution functions to determine halo and
    stellar particle velocities in initial conditions.

    Parameters
    ----------
    model : :class:`~cluster_generator.model.ClusterModel`
        The cluster model which will be used to
        construct the virial equilibrium.
    ptype : string, optional
        The type of the particles which can be generated from this
        object, either "dark_matter" or "stellar". Default: "dark_matter"
    df : unyt_array
        The particle distribution function. If not supplied, it will
        be generated.
    gravity: str
        .. warning::
            Work in progress.
    type: str
        The type of equilibrium to facilitate. If ``eddington``, the edington formula is used. if ``lma``, then LMA is used.
    Notes
    -----

    Generating the Phase Space Density
    ++++++++++++++++++++++++++++++++++
    The particles in the simulation may be thought of as occupying a 6-dimensional "phase space", :math:`\textbf{x}\times\dot{\textbf{x}}`.
    Regardless of the dynamics of the system, Liouville's Theorem applies to the phase space density of the system and therefore,
    the Collisionless Boltzmann Equation and the Jean's Equation are both valid universally. The question of virialization is to
    determine the velocities of the constituent particles in such a way that the system remains in a steady state.

    **The Eddington Formula Approach**:

    .. attention ::

        This approach is only viable with Newtonian gravity. The integrand generated in this process is only invertible for
        the phase density because it fits the form of an Abel Integral, which generically fails to happen in non-newtonian
        gravity theories.

    Consider a velocity distribution function :math:`f(\textbf{v})`. Then (in Newtonian gravity),

    .. math::

        \nabla ^2 \Phi = 4\pi G \rho = 4\pi G \int f d^3\textbf{v}.

    The RHS of the equation may be expanded in spherical coordinates to yield a non-trivial differential equation in :math:`f`.
    In the case of galaxy clusters, the Jean's Theorem applies, stipulating that :math:`f` be a function only of energy and angular momentum. If,
    as is typical, we assume that the system is irrotational, the angular momentum is constant and the function becomes dependent only
    on the energy. To simplify the notation, we denote the **relative potential** to be :math:`\Psi = -\Phi + \Phi_0` and the relative energy
    to be :math:`\mathcal{E} = -E + \Phi_0 = \Psi - \frac{1}{2}v^2`.  In this case, the equation above may be simplified to the form

    .. math::

        \rho(r) = 4\pi \int_0^\Psi f(\mathcal{E})\sqrt{2(\Psi - \mathcal{E})} d\mathcal{E}.

    If :math:`\rho` is considered a function of the relative potential, then one may obtain the equation

    .. math::

        \frac{1}{\sqrt{8}\pi}\frac{d\rho}{d\Psi} = \int_0^\Psi \frac{f(\mathcal{E}) d\mathcal{E}}{\sqrt{\Psi-\mathcal{E}}}

    Blinney and Tremaine note that this is an Abel integral and may be inverted to yield

    .. math::

        f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2} \frac{d}{d\mathcal{E}}\int_0^\mathcal{E} \frac{d\rho}{d\Psi} \frac{d\Psi}{\sqrt{\mathcal{E}-\Psi}}

    This is the approach used in ``cluster_generator`` for the generation of virialized systems when using Newtonian dynamics.

    **The Local Maxwellian Approximation**

    While the Eddington Formula provides an ideal approach in Newtonian gravity, when using non-newtonian gravity or badly behaved profiles even in Newtonian
    gravity, analytical approaches may become non-viable; however, the Local Maxwellian approximation (LMA) can be used to obtain a viable estimate
    for the distribution function nonetheless. Consider the Jeans Equation in spherically symmetry:

    .. math::

        \frac{\partial \rho \sigma_r^2}{\partial r} + \frac{2\rho}{r}\left(\sigma_r^2 -\sigma^2_\theta\right) + \rho \frac{\partial \Phi}{\partial r} = 0.

    If one also assumes that the stress tensor is entirely isotropic, then the equation may be manipulated to allow one to find the velocity dispersion

    .. math::

        \sigma_r^2 = \frac{1}{\rho_h}\int_r^\infty \rho_h \frac{d\Phi}{dr} dr

    We are then able to assume a gaussian distribution function for the velocities, and thus a maxwellian distribution for the speeds

    .. math::

        F(v,r) = 4\pi \left(\frac{1}{2\pi \sigma^2}\right) v^2 \exp\left(\frac{-v^2}{2\bar{v_r}^2}\right)

    """

    #  Dunder Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    def __init__(self, model, ptype='dark_matter', df=None,sigma2=None,type="eddington"):
        #  Assigning basic attributes
        # ------------------------------------------------------------------------------------------------------------ #
        #: The number of elements in the model's field samples. (Inherited)
        self.num_elements = model.num_elements
        #: the particle type (``dark_matter`` or ``stellar``).
        self.ptype = ptype
        #: The model object corresponding to the virialization problem.
        self.model = model
        #: The type of equilibrium construction to use.
        self.type = type

        if self.type == "eddington":
            self._df,self._sigma = None, None
            # - Generate the DF function - #
            if df is None:
                _ = self.df
            else:
                self._df = df

        elif self.type == "lma":
            self._df,self._sigma = None, None
            if sigma2 is None:
                _ = self.sigma
            else:
                self._sigma = sigma2

        else:
            raise ValueError(f"The equilibrium type {self.type} is not valid.")

    # Properties
    # ----------------------------------------------------------------------------------------------------------------- #
    @property
    def df(self):
        if self.type == "eddington":
            if self._df is not None:
                return self._df
            else:
                self._generate_df()
        else:
            mylog.warning(f"{self.__repr__()} has equilibrium type {self.type} and so no 'df' exists.")

    @df.setter
    def df(self,value):
        if self.type == "eddington":
            self._df = value
            f = value.d[::-1]
            self.f = InterpolatedUnivariateSpline(self.ee,f)
        else:
            mylog.warning(f"{self.__repr__()} has equilibrium type {self.type} and so no 'df' exists.")

    @property
    def sigma(self):
        if self.type == "lma":
            if self._sigma is not None:
                return self._sigma
            else:
                self._generate_sigma()
        else:
            mylog.warning(f"{self.__repr__()} has equilibrium type {self.type} and so no 'sigma' exists.")

    @sigma.setter
    def sigma(self,value):
        if self.type == "lma":
            self._sigma = value
        else:
            mylog.warning(f"{self.__repr__()} has equilibrium type {self.type} and so no 'sigma' exists.")

    @property
    def ee(self):
        """
        The ``relative potential``, :math:`\Psi(r) = -\Phi +\Phi_0`. In this case, :math:`\Phi_0 = 0`.

        .. warning::

            The relative potential here is in order of decreasing :math:`r`.
        Returns
        -------

        """
        return -self.model["gravitational_potential"].d[::-1]

    @property
    def ff(self):
        """Reverse direction of ``self.df`` as a numpy array. """
        return self._df.d[::-1]

    #  Private Functions
    # ----------------------------------------------------------------------------------------------------------------- #
    def _generate_df(self):
        """
        Generates the distribution function :math:`DF(\textbf{x},\textbf{v}` for the given cluster.

        Returns
        -------

        """
        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if self.type != "eddington":
            raise ValueError(f"The method _generate_df is not available to VirialEquilibrium models with type {self.type}")
        #  Pulling necessary field data out of the model.
        # ------------------------------------------------------------------------------------------------------------ #
        # --- Particle density --- #
        # ? Grabs the particle densities and then constructs a density spline as a function of the
        # ? relative potential (self.ee)
        pden = self.model[f"{self.ptype}_density"][::-1]
        density_spline = InterpolatedUnivariateSpline(self.ee, pden)

        # -- Preparing to conduct the integration -- #
        g = np.zeros(self.num_elements)
        dgdp = lambda t, e: 2*density_spline(e-t*t, 1)

        pbar = tqdm(leave=True, total=self.num_elements,
                    desc="Computing particle DF (Eddington) ")
        for i in range(self.num_elements):
            g[i] = quad(dgdp, 0., np.sqrt(self.ee[i]), epsabs=1.49e-05,
                        epsrel=1.49e-05, args=(self.ee[i]))[0]
            pbar.update()
        pbar.close()

        g_spline = InterpolatedUnivariateSpline(self.ee, g)
        ff = g_spline(self.ee, 1)/(np.sqrt(8.)*np.pi**2)
        self.f = InterpolatedUnivariateSpline(self.ee, ff)
        self._df = unyt_array(ff[::-1], "Msun*Myr**3/kpc**6")

    def _generate_sigma(self):
        """
        Generates the :math:`\sigma^2` values for the distribution function at each radius.

        Returns
        ------
        """
        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        # - Checking type - #
        if self.type != "lma":
            raise ValueError(
                f"The method _generate_sigma is not available to VirialEquilibrium models with type {self.type}")

        # - Asserting necessary fields - #
        if self.model["gravitational_potential"] is None:
            _ = self.model.pot # Generates the potential from the inherited model.

        #  Constructing splines for estimation
        # ------------------------------------------------------------------------------------------------------------ #
        # ! DENSITY SPLINE: truncation approach to force asymptotic behavior at large radii.
        #
        # - Generating the cutoff radius - #
        _rmax = np.amax(self.model["radius"].d)

        # - Building the density spline - #
        pden = self.model[f"{self.ptype}_density"].to("Msun/kpc**3")
        density_spline = InterpolatedUnivariateSpline(self.model["radius"].d, pden.d,k=3)

        _slope = _rmax*density_spline(_rmax,1)/density_spline(_rmax)
        _density_spline = lambda r: np.piecewise(r,[r<_rmax,r>=_rmax],
                                                 [density_spline,lambda l: density_spline(_rmax)*(l/_rmax)**_slope])

        # ! POTENTIAL SPLINE: truncating slope on the integrand at large radii.

        # - Potential - #
        g = self.model["gravitational_potential"].to("kpc**2/Myr**2")
        potential_spline = InterpolatedUnivariateSpline(self.model["radius"].d,g.d,k=3)
        _dp_func = lambda r: np.piecewise(r,[r<_rmax,r>=_rmax],
                                          [lambda l: potential_spline(l,1),lambda l: potential_spline(_rmax,1)])

        # --- Building the integrands --- #
        integrand = lambda x: density_spline(x)*potential_spline(x,1)
        inf_integrand = lambda x: _density_spline(x)*_dp_func(x)

        #  Performing the integration
        # ------------------------------------------------------------------------------------------------------------ #
        sig = np.zeros(self.num_elements)
        pbar = tqdm(leave=True, total=self.num_elements,
                    desc="Computing particle dispersions (LMA) ")

        inf_int = quad(inf_integrand,_rmax,np.inf,limit=100,epsabs=1.49e-05,
                        epsrel=1.49e-05)[0]

        for i,r in enumerate(self.model["radius"].d):
            sig[i] = quad(integrand, r,_rmax,limit=100, epsabs=1.49e-05,
                        epsrel=1.49e-05)[0]
            pbar.update()
        pbar.close()
        sig += inf_int
        self._sigma = unyt_array(sig,"Msun / (Myr**2 * kpc)")/pden
        self._sigma = self._sigma.to("kpc**2/Myr**2")

    def check_virial(self):
        r"""
        Computes the radial density profile for the collisionless 
        particles computed from integrating over the distribution 
        function, and the relative difference between this and the 
        input density profile.

        Returns
        -------
        rho : NumPy array
            The density profile computed from integrating the
            distribution function. 
        chk : NumPy array
            The relative difference between the input density
            profile and the one calculated using this method.

        Notes
        -----
        See [1]_ for a complete discussion. The basis of the derivation is as follows:

        Let :math:`f(\textbf{v})` be the distribution function of the system. Then, by definition,

        .. math::

            \rho =  \int f d^3\textbf{v}.

        Invoking the Jeans Theorem, we may acknowledge that :math:`f` is a function only of :math:`L,\mathcal{E}`. Furthermore,
        due to the assumption of an isotropic dispersion tensor, :math:`f` is a function only of the energy. We therefore write
        the above equation as

        .. math::

                \rho = 4\pi \int_0^\Psi f(\mathcal{E})\sqrt{2(\Psi-\mathcal{E})}\; d\mathcal{E},

        where
        .. math::

            \Psi = -\Phi + \Phi_0, \;\;\text{and}\;\;\mathcal{E} = \Psi - \frac{1}{2}v^2.

        These quantities are referred to as **relative potential** and **relative energy**.

        References
        ----------
            [1] Binney, J., & Tremaine, S. (2011). Galactic dynamics (Vol. 20). Princeton university press.
        """
        #  Preparing arrays and pulling data
        # ----------------------------------------------------------------------------------------------------------------- #
        if self.type != "eddington":
            raise ValueError("The model does not use a form of virialization that can use this function.")
        n = self.num_elements
        rho = np.zeros(n)
        pden = self.model[f"{self.ptype}_density"].d #-> This is the profile / model density array.

        # - Defining the integrand - #
        rho_int = lambda e, psi: self.f(e)*np.sqrt(2*(psi-e))

        # - Carrying out the cumulative integration - #
        for i, e in enumerate(self.ee):
            rho[i] = 4.*np.pi*quad(rho_int, 0., e, args=(e,))[0]

        # - performing the check.
        chk = (rho[::-1]-pden)/pden
        mylog.info("The maximum relative deviation of this profile from "
                   "virial equilibrium is %g", np.abs(chk).max())
        return rho[::-1], chk

    def generate_particles(self, num_particles, r_max=None, sub_sample=1,
                           compute_potential=False, prng=None):
        """
        Generate a set of dark matter or star particles in virial equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate 
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii and velocities which will then be repeated
            to fill the required number of particles. Default: 1, which
            means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only 
            be specified if you have a reason to generate the same 
            set of random numbers, such as for a test. Default is None, 
            which sets the seed based on the system time.

        Returns
        -------
        particles : :class:`~cluster_generator.particles.ClusterParticles`
            A set of dark matter or star particles.
        """
        #  Setup
        # ----------------------------------------------------------------------------------------------------------------- #
        from cluster_generator.utils import parse_prng

        num_particles_sub = num_particles // sub_sample # number of particles for which to generate.
        key = {"dark_matter": "dm",  "stellar": "star"}[self.ptype] # particle type reference.

        #- Pulling fields - #
        density = f"{self.ptype}_density"
        mass = f"{self.ptype}_mass"

        #  Constructing the particle positions
        # ----------------------------------------------------------------------------------------------------------------- #
        prng = parse_prng(prng)

        mylog.info("We will be assigning %s %s particles.",
                   num_particles, self.ptype)
        mylog.info("Compute %s particle positions.", num_particles)

        # --- Determining the appropriate radii by inverse sampling --- #
        nonzero = self.model[density] > 0.0
        radius_sub, mtot = generate_particle_radii(self.model["radius"].d[nonzero],
                                                   self.model[mass].d[nonzero],
                                                   num_particles_sub, r_max=r_max,
                                                   prng=prng)

        # --- Tiling if there is a sub-sampling procedure --- #
        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        # --- Angular Distribution (uniform) --- #
        theta = np.arccos(prng.uniform(low=-1., high=1., size=num_particles))
        phi = 2.*np.pi*prng.uniform(size=num_particles)

        fields = OrderedDict()

        fields[key, "particle_position"] = unyt_array(
            [radius*np.sin(theta)*np.cos(phi), radius*np.sin(theta)*np.sin(phi),
             radius*np.cos(theta)], "kpc").T

        # Computing necessary velocities
        # ------------------------------------------------------------------------------------------------------------ #
        if self.type == "eddington":
            mylog.info(f"Computing {self.ptype} velocities using {self.type} method.")

            #  Solving Via the Eddington Formula
            # -------------------------------------------------------------------------------------------------------- #

            # --- Pulling the necessary fields --- #
            energy_spline = InterpolatedUnivariateSpline(self.model["radius"].d,
                                                         self.ee[::-1])

            psi = energy_spline(radius_sub) # - Relative potential pulled from energy spline function.
            vesc = 2.*psi # - (after root) becomes the escape velocity.
            fv2esc = vesc*self.f(psi) # - self.f is the distribution function (backward),
            vesc = np.sqrt(vesc)

            velocity_sub = generate_velocities(
                psi, vesc, fv2esc, self.f._eval_args[0], self.f._eval_args[1],
                self.f._eval_args[2])


        elif self.type == "lma":
            mylog.info(f"Computing {self.ptype} velocities using {self.type} method.")

            #  Solving via the local maxwellian approximation
            # -------------------------------------------------------------------------------------------------------- #

            # --- Generating the dispersion spline --- #
            dispersion_spline = InterpolatedUnivariateSpline(self.model["radius"],self.sigma)
            import matplotlib.pyplot as plt
            plt.loglog(self.model["radius"],dispersion_spline(self.model["radius"]))
            plt.show()
            dispersion_array = dispersion_spline(radius_sub) # assigns the correct dispersion to each of the radii in the particle sample.

            # --- Computing the relevant escape velocities at each radius --- #
            potential_spline = InterpolatedUnivariateSpline(self.model["radius"],self.model.pot)
            _potentials = potential_spline(radius_sub)
            vesc = np.sqrt(2*np.abs(_potentials))

            # --- Creating the distribution function --- #
            _base_array = np.linspace(0,3,1000)
            _cumval = erf(_base_array) - (2/np.sqrt(np.pi))*_base_array*np.exp(-_base_array**2)

            cumdist_spline = InterpolatedUnivariateSpline(_cumval,_base_array,ext=0)

            # --- Calling the generator --- #
            velocity_sub = generate_lma_velocities(
                dispersion_array,
                vesc,
                cumdist_spline._eval_args[0],
                cumdist_spline._eval_args[1],
                cumdist_spline._eval_args[2],
                0.950
            )


        else:
            raise ValueError(f"The virialization method {self.type} was not recognized.")

        #  Cleanup
        # ------------------------------------------------------------------------------------------------------------ #

        if sub_sample > 1:
            velocity = np.tile(velocity_sub, sub_sample)[:num_particles]
        else:
            velocity = velocity_sub

        theta = np.arccos(prng.uniform(low=-1., high=1., size=num_particles))
        phi = 2. * np.pi * prng.uniform(size=num_particles)

        fields[key, "particle_velocity"] = unyt_array(
            [velocity * np.sin(theta) * np.cos(phi),
             velocity * np.sin(theta) * np.sin(phi),
             velocity * np.cos(theta)], "kpc/Myr").T

        fields[key, "particle_mass"] = unyt_array(
            [mtot / num_particles] * num_particles, "Msun")

        if compute_potential:
            if self.type == "eddington":
                if sub_sample > 1:
                    phi = -np.tile(psi, sub_sample)
                else:
                    phi = -psi
            else:
                if sub_sample > 1:
                    phi = np.tile(_potentials,sub_sample)
                else:
                    phi = _potentials

            fields[key, "particle_potential"] = unyt_array(
                phi, "kpc**2/Myr**2")

        return ClusterParticles(key, fields)
