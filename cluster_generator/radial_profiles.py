"""
===============
Radial Profiles
===============
Radial profiles, as instantiated through the :py:class:`radial_profiles.RadialProfile` class are wrappers for standard ``lambda`` functions
which are used to provide additional structure for radial profiles of temperature, density, cumulative mass, and entropy in galaxy clusters.

Each ``RadialProfile`` object has attached methods for altering the profile to add cores, truncate the profile, and a variety of other tasks.

"""
import numpy as np

#  Minor Functions
# ----------------------------------------------------------------------------------------------------------------- #
#: Alternative factor for rho(r) in NFW profiles. See `the wiki<https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile>`_
_nfw_factor = lambda conc: 1.0/(np.log(conc+1.0)-conc/(1.0+conc))


#  Classes
# ----------------------------------------------------------------------------------------------------------------- #
class RadialProfile:
    r"""
    The ``RadialProfile`` class is a container class for all of the radial profiles in ``cluster_generator``.

    Parameters
    ----------
    profile: RadialProfile or callable
        The radial profile to attribute to the object. The radial profile must be callable (i.e. ``lambda`` function) or
        another instance of ``RadialProfile``.

        .. admonition:: info

            If another instance of ``RadialProfile`` is passed, the ``RadialProfile.profile`` object is passed so that
            the new ``RadialProfile`` object has the same profile as the previous one.

    """
    def __init__(self, profile):
        if isinstance(profile, RadialProfile):
            # Consistency check for profile type consistency.
            self.profile = profile.profile
        else:
            self.profile = profile

    def __call__(self, r):
        return self.profile(r)

    def _do_op(self, other, op):
        # Allows for operations between profiles.
        if hasattr(other, "profile"):
            p = lambda r: op(self.profile(r), other.profile(r))
        else:
            p = lambda r: op(self.profile(r), other)
        return p

    def __add__(self, other):
        p = self._do_op(other, np.add)
        return RadialProfile(p)

    def __mul__(self, other):
        p = self._do_op(other, np.multiply)
        return RadialProfile(p)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, power):
        p = lambda r: self.profile(r)**power
        return RadialProfile(p)

    def add_core(self, r_core, alpha):
        r"""
        Adds a core to the pre-existing profile.

        Parameters
        ----------
        r_core : float
            The core radius in kpc.
        alpha : float
            The power-low index inside the exponential.

        Notes
        -----
        ``add_core`` is implemented by taking the existing profile :math:`f(r)` and altering it such that

        .. math::

            f'(r) = \left(1-\exp\left(\frac{-r}{r_{core}}\right)^\alpha\right) f(r).

        This will cause any cuspy profile (i.e. one for which :math:`\left.\frac{d}{dr} f(r)\right|_{r=0} > 0` and which grows
        faster than the exponential term added to instead contain a core and go to 0 in its limit.
        """
        def _core(r):
            x = r/r_core
            ret = 1.0-np.exp(-x**alpha)
            return self.profile(r)*ret
        return RadialProfile(_core)

    def cutoff(self, r_cut, k=5):
        r"""
        Generates a truncated form of the profile.

        Parameters
        ----------
        r_cut: float or int
            The cutoff radius beyond which the truncation should dominate the profile behavior [kpc].
        k: int
            The truncation rate. Higher ``k`` will cause the truncation to go to zero faster.

        Returns
        -------
        RadialProfile
            The corresponding ``RadialProfile`` object with the truncated profile.

        Notes
        -----
        The truncation is achieved by multiplying the profile by the factor

        .. math::

            1-\frac{1}{1+\exp\left(-2k\left(\frac{r}{r_{cut}}\right)\right)}.
        """
        def _cutoff(r):
            x = r/r_cut
            step = 1.0/(1.0+np.exp(-2*k*(x-1)))
            p = self.profile(r)*(1.0-step)
            return p
        return RadialProfile(_cutoff)

    @classmethod
    def from_array(cls, r, f_r):
        """
        Generate a callable radial profile using an array of radii
        and an array of values. 

        Parameters
        ----------
        r : array-like
            Array of radii in kpc.
        f_r : array-like
            Array of profile values in the appropriate units.

        Returns
        -------
        RadialProfile
            The corresponding radial profile.

        Notes
        -----
        This function uses ``scipy.interpolate.UnivariateSpline`` to generate a continuous spectrum.
        """
        from scipy.interpolate import UnivariateSpline
        f = UnivariateSpline(r, f_r)
        return cls(f)

    @classmethod
    def from_binary(cls,f):
        """
        Loads a specific instance of a ``RadialProfile`` object from the serialized version of the instance saved to disk.

        Parameters
        ----------
        f: str
            The filename to open. Should be a valid ``.rp`` file type.

        Returns
        -------
        RadialProfile
            The ``RadialProfile`` object on disk.

        Notes
        -----
        The serialization of the ``RadialProfile`` object is done using the ``pickle`` library.
        """
        import dill as pickle
        with open(f,"rb") as bf:
            return pickle.load(bf)

    def to_binary(self,f):
        """
        Sends the ``RadialProfile`` instance to a serialized binary file.

        Parameters
        ----------
        f: str
            The preferred filename. For consistency, binary files should have ``.rp`` extension; however, this is not required.

        Returns
        -------
        None

        """
        import dill as pickle
        with open(f,"wb") as bf:

            pickle.dump(self,bf)

    def plot(self, rmin, rmax, num_points=1000, fig=None, ax=None,
             lw=2, **kwargs):
        """
        Make a quick plot of a profile using Matplotlib.

        Parameters
        ----------
        rmin : float
            The minimum radius of the plot in kpc.
        rmax : float
            The maximum radius of the plot in kpc. 
        num_points : integer, optional
            The number of logspaced points between rmin
            and rmax to use when making the plot. Default: 1000
        fig : :class:`~matplotlib.figure.Figure`, optional
            A Figure instance to plot in. Default: None, one will be
            created if not provided.
        ax : :class:`~matplotlib.axes.Axes`, optional
            An Axes instance to plot in. Default: None, one will be
            created if not provided.
        """
        import matplotlib.pyplot as plt
        plt.rc("font", size=18)
        plt.rc("axes", linewidth=2)
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        if ax is None:
            ax = fig.add_subplot(111)
        rr = np.logspace(np.log10(rmin), np.log10(rmax),
                         num_points, endpoint=True)
        ax.loglog(rr, self(rr), lw=lw, **kwargs)
        ax.set_xlabel("Radius (kpc)")
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        return fig, ax


def constant_profile(const):
    """
    A constant profile.

    Parameters
    ----------
    const : float
        The value of the constant.
    """
    p = lambda r: const
    return RadialProfile(p)


def power_law_profile(A, r_s, alpha):
    """
    A profile which is a power-law with radius, scaled
    so that it has a certain value ``A`` at a scale 
    radius ``r_s``. Can be used as a density, temperature,
    mass, or entropy profile (or whatever else one may
    need).

    Parameters
    ----------
    A : float
        Scale value of the profile at r = r_s.
    r_s : float
        Scale radius in kpc.
    alpha : float
        Power-law index of the profile.
    """
    p = lambda r: A*(r/r_s)**alpha
    return RadialProfile(p)


def beta_model_profile(rho_c, r_c, beta):
    """
    A beta-model density profile [1]_.

    Parameters
    ----------
    rho_c : float
        The core density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    beta : float
        The beta parameter.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [1] (Cavaliere A.,Fusco-Femiano R., 1976, A&A, 49, 137).
    """
    p = lambda r: rho_c*((1+(r/r_c)**2)**(-1.5*beta))
    return RadialProfile(p)


def hernquist_density_profile(M_0, a):
    """
    A Hernquist density profile [1].

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [1] (Hernquist, L. 1990, ApJ, 356, 359).
    """
    p = lambda r: M_0/(2.*np.pi*a**3)/((r/a)*(1.+r/a)**3)
    return RadialProfile(p)


def cored_hernquist_density_profile(M_0, a, b):
    """
    A Hernquist density profile (Hernquist, L. 1990,
    ApJ, 356, 359) with a core radius.

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    b : float
        The core radius in kpc.
    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    p = lambda r: M_0*b/(2.*np.pi*a**3)/((1.+b*r/a)*(1.+r/a)**3)
    return RadialProfile(p)


def hernquist_mass_profile(M_0, a):
    """
    A Hernquist mass profile (Hernquist, L. 1990,
    ApJ, 356, 359).

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    p = lambda r: M_0*r**2/(r+a)**2
    return RadialProfile(p)


def convert_nfw_to_hernquist(M_200, r_200, conc):
    """
    Given M200, r200, and a concentration parameter for an
    NFW profile, return the Hernquist mass and scale radius
    parameters.

    Parameters
    ----------
    M_200 : float
        The mass of the halo at r200 in Msun.
    r_200 : float
        The radius corresponding to the overdensity of 200 times the
        critical density of the universe in kpc.
    conc : float
        The concentration parameter r200/r_s for the NFW profile.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    a = r_200/(np.sqrt(0.5*conc*conc*_nfw_factor(conc))-1.0)
    M0 = M_200*(r_200+a)**2/r_200**2
    return M0, a


def nfw_density_profile(rho_s, r_s):
    """
    An NFW density profile (Navarro, J.F., Frenk, C.S.,
    & White, S.D.M. 1996, ApJ, 462, 563).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    p = lambda r: rho_s/((r/r_s)*(1.0+r/r_s)**2)
    return RadialProfile(p)


def nfw_mass_profile(rho_s, r_s):
    """
    An NFW mass profile (Navarro, J.F., Frenk, C.S.,
    & White, S.D.M. 1996, ApJ, 462, 563).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    def _nfw(r):
        x = r/r_s
        return 4*np.pi*rho_s*r_s**3*(np.log(1+x)-x/(1+x))
    return RadialProfile(_nfw)


def nfw_scale_density(conc, z=0.0, delta=200.0, cosmo=None):
    """
    Compute a scale density parameter for an NFW profile
    given a concentration parameter, and optionally
    a redshift, overdensity, and cosmology.

    Parameters
    ----------
    conc : float
        The concentration parameter for the halo, which should 
        correspond the selected overdensity (which has a default
        of 200). 
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    delta : float, optional
        The overdensity parameter for which the concentration
        is defined. Default: 200.0
    cosmo : yt Cosmology object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will 
        be used.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    from yt.utilities.cosmology import Cosmology
    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    rho_s = delta*rho_crit*conc**3*_nfw_factor(conc)/3.
    return rho_s


def tnfw_density_profile(rho_s, r_s, r_t):
    """
    A truncated NFW (tNFW) density profile (Baltz, E.A.,
    Marshall, P., & Oguri, M. 2009, JCAP, 2009, 015).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    def _tnfw(r):
        profile = rho_s/((r/r_s)*(1+r/r_s)**2)
        profile /= (1+(r/r_t)**2)
        return profile
    return RadialProfile(_tnfw)


def tnfw_mass_profile(rho_s, r_s, r_t):
    """
    A truncated NFW (tNFW) mass profile (Baltz, E.A.,
    Marshall, P., & Oguri, M. 2009, JCAP, 2009, 015).

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    from sympy import Symbol, integrate, lambdify
    xx = Symbol("x")
    aa = Symbol("a")
    yy = Symbol("y")
    f = integrate(xx**2/(xx*(1+xx)**2)/(1+(xx/aa)**2), (xx, 0, yy))
    fl = lambdify((yy, aa), f, modules="numpy")
    def _tnfw(r):
        x = r / r_s
        a = r_t / r_s
        return 4*np.pi*rho_s*r_s**3*fl(x, a).astype('float64')
    return RadialProfile(_tnfw)


def snfw_density_profile(M, a):
    """
    A "super-NFW" density profile (Lilley, E. J.,
    Wyn Evans, N., & Sanders, J.L. 2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    def _snfw(r):
        x = r/a
        return 3.*M/(16.*np.pi*a**3)/(x*(1.+x)**2.5)
    return RadialProfile(_snfw)


def snfw_mass_profile(M, a):
    """
    A "super-NFW" mass profile (Lilley, E. J.,
    Wyn Evans, N., & Sanders, J.L. 2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    def _snfw(r):
        x = r/a
        return M*(1.-(2.+3.*x)/(2.*(1.+x)**1.5))
    return RadialProfile(_snfw)


def snfw_total_mass(mass, radius, a):
    """
    Find the total mass parameter for the super-NFW
    model by inputting a reference mass and radius 
    (say, M200c and R200c), along with the scale radius.

    Parameters
    ----------
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    mp = snfw_mass_profile(1.0, a)
    return mass/mp(radius)


def cored_snfw_density_profile(M, a, r_c):
    """
    A cored "super-NFW" density profile (Lilley, E. J.,
    Wyn Evans, N., & Sanders, J.L. 2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    b = a/r_c
    def _snfw(r):
        x = r/a
        return 3.*M*b/(16.*np.pi*a**3)/((1.+b*x)*(1.+x)**2.5)
    return RadialProfile(_snfw)


def cored_snfw_mass_profile(M, a, r_c):
    """
    A cored "super-NFW" mass profile (Lilley, E. J.,
    Wyn Evans, N., & Sanders, J.L. 2018, MNRAS).

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    b = a/r_c
    def _snfw(r):
        x = r/a
        y = np.complex128(np.sqrt(x+1.))
        d = np.sqrt(np.complex128(b/(1.0-b)))
        e = b*(b-1.)**2
        ret = (1.0-1.0/y)*(b-2.)/(b-1.)**2
        ret += (1.0/y**3-1.0)/(3.*(b-1.))
        ret += d*(np.arctan(y*d)-np.arctan(d))/e
        return 1.5*M*b*ret.astype("float64")
    return RadialProfile(_snfw)


def snfw_conc(conc_nfw):
    """
    Given an NFW concentration parameter, calculate the 
    corresponding sNFW concentration parameter. This comes
    from Equation 31 of (Lilley, E. J., Wyn Evans, N., & 
    Sanders, J.L. 2018, MNRAS).

    Parameters
    ----------
    conc_nfw : float
        NFW concentration for r200c.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    return 0.76*conc_nfw+1.36


def cored_snfw_total_mass(mass, radius, a, r_c):
    """
    Find the total mass parameter for the cored super-NFW
    model by inputting a reference mass and radius 
    (say, M200c and R200c), along with the scale radius.

    Parameters
    ----------
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    mp = cored_snfw_mass_profile(1.0, a, r_c)
    return mass/mp(radius)


_dn = lambda n: 3.0*n - 1./3. + 8.0/(1215.*n) + 184.0/(229635.*n*n)


def einasto_density_profile(M, r_s, n):
    """
    A density profile where the logarithmic slope is a 
    power-law. The form here is that given in Section 2 of
    Retana-Montenegro et al. 2012, A&A, 540, A70.

    Parameters
    ----------
    M : float
        The total mass of the profile in M.
    r_s : float
        The scale radius in kpc.
    n : float
        The inverse power-law index.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    from scipy.special import gamma
    alpha = 1.0/n
    h = r_s/_dn(n)**n
    rho_0 = M/(4.0*np.pi*h**3*n*gamma(3.0*n))
    def _einasto(r):
        s = r/h
        return rho_0*np.exp(-s**alpha)
    return RadialProfile(_einasto)


def einasto_mass_profile(M, r_s, n):
    """
    A mass profile where the logarithmic slope is a 
    power-law. The form here is that given in Section 2 of
    Retana-Montenegro et al. 2012, A&A, 540, A70.

    Parameters
    ----------
    M : float
        The total mass of the profile in M.
    r_s : float
        The scale radius in kpc.
    n : float
        The inverse power-law index.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    from scipy.special import gammaincc
    alpha = 1.0/n
    h = r_s/_dn(n)**n
    def _einasto(r):
        s = r/h
        return M*(1.0-gammaincc(3.0*n, s**alpha))
    return RadialProfile(_einasto)


def am06_density_profile(rho_0, a, a_c, c, n):
    """
    The density profile for galaxy clusters suggested by
    Ascasibar, Y., & Markevitch, M. 2006, ApJ, 650, 102.
    Works best in concert with the ``am06_temperature_profile``.

    Parameters
    ----------
    rho_0 : float
        The scale density of the profile in Msun/kpc**3.
    a : float
        The scale radius in kpc.
    a_c : float
        The scale radius of the cool core in kpc.
    c : float
        The scale of the temperature drop of the cool core.
    n : float

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    alpha = -1.-n*(c-1.)/(c-a/a_c)
    beta = 1.-n*(1.-a/a_c)/(c-a/a_c)
    p = lambda r: rho_0*(1.+r/a_c)*(1.+r/a_c/c)**alpha*(1.+r/a)**beta
    return RadialProfile(p)


def vikhlinin_density_profile(rho_0, r_c, r_s, alpha, beta,
                              epsilon, gamma=None):
    """
    A modified beta-model density profile for galaxy
    clusters from Vikhlinin, A., Kravtsov, A., Forman, W.,
    et al. 2006, ApJ, 640, 691.

    Parameters
    ----------
    rho_0 : float
        The scale density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    r_s : float
        The scale radius in kpc.
    alpha : float
        The inner logarithmic slope parameter.
    beta : float
        The middle logarithmic slope parameter.
    epsilon : float
        The outer logarithmic slope parameter.
    gamma : float
        This parameter controls the width of the outer
        transition. If None, it will be gamma = 3 by default.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    if gamma is None:
        gamma = 3.0
    profile = lambda r: rho_0*(r/r_c)**(-0.5*alpha) * \
                        (1.+(r/r_c)**2)**(-1.5*beta+0.25*alpha) * \
                        (1.+(r/r_s)**gamma)**(-0.5*epsilon/gamma)
    return RadialProfile(profile)


def vikhlinin_temperature_profile(T_0, a, b, c, r_t, T_min,
                                  r_cool, a_cool):
    """
    A temperature profile for galaxy clusters from
    Vikhlinin, A., Kravtsov, A., Forman, W., et al.
    2006, ApJ, 640, 691.

    Parameters
    ----------
    T_0 : float
        The scale temperature of the profile in keV.
    a : float
        The inner logarithmic slope.
    b : float
        The width of the transition region.
    c : float
        The outer logarithmic slope.
    r_t : float
        The scale radius kpc.
    T_min : float
        The minimum temperature in keV.
    r_cool : float
        The cooling radius in kpc.
    a_cool : float
        The logarithmic slope in the cooling region.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    def _temp(r):
        x = (r/r_cool)**a_cool
        t = (r/r_t)**(-a)/((1.+(r/r_t)**b)**(c/b))
        return T_0*t*(x+T_min/T_0)/(x+1)
    return RadialProfile(_temp)


def am06_temperature_profile(T_0, a, a_c, c):
    """
    The temperature profile for galaxy clusters suggested by
    Ascasibar, Y., & Markevitch, M. 2006, ApJ, 650, 102.
    Works best in concert with the ``am06_density_profile``.

    Parameters
    ----------
    T_0 : float
        The scale temperature of the profile in keV.
    a : float
        The scale radius in kpc.
    a_c : float
        The cooling radius in kpc.
    c : float
        The scale of the temperature drop of the cool core.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    p = lambda r: T_0/(1.+r/a)*(c+r/a_c)/(1.+r/a_c)
    return RadialProfile(p)


def baseline_entropy_profile(K_0, K_200, r_200, alpha):
    """
    The baseline entropy profile for galaxy clusters (Voit, G.M.,
    Kay, S.T., & Bryan, G.L. 2005, MNRAS, 364, 909).

    Parameters
    ----------
    K_0 : float
        The central entropy floor in keV*cm**2.
    K_200 : float
        The entropy at the radius r_200 in keV*cm**2.
    r_200 : float
        The virial radius in kpc.
    alpha : float
        The logarithmic slope of the profile.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.
    """
    p = lambda r: K_0 + K_200*(r/r_200)**alpha
    return RadialProfile(p)


def broken_entropy_profile(r_s, K_scale, alpha, K_0=0.0):
    def _entr(r):
        x = r/r_s
        ret = (x**alpha)*(1.+x**5)**(0.2*(1.1-alpha))
        return K_scale*(K_0+ret)
    return RadialProfile(_entr)


def walker_entropy_profile(r_200, A, B, K_scale, alpha=1.1):
    def _entr(r):
        x = r/r_200
        return K_scale*(A*x**alpha)*np.exp(-(x/B)**2)
    return RadialProfile(_entr)


def rescale_profile_by_mass(profile, mass, radius):
    """
    Rescale a density ``profile`` by a total ``mass``
    within some ``radius``.

    Parameters
    ----------
    profile : ``RadialProfile`` object
        The profile object to rescale.
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.

    """
    from scipy.integrate import quad
    mass_int = lambda r: profile(r)*r*r
    rescale = mass/(4.*np.pi*quad(mass_int, 0.0, radius)[0])
    return rescale*profile


def find_overdensity_radius(m, delta, z=0.0, cosmo=None):
    """
    Given a mass value and an overdensity, find the radius
    that corresponds to that enclosed mass.

    Parameters
    ----------
    m : float
        The enclosed mass.
    delta : float
        The overdensity to compute the radius for.
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    cosmo : yt ``Cosmology`` object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will 
        be used.
    """
    from yt.utilities.cosmology import Cosmology
    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    return (3.0*m/(4.0*np.pi*delta*rho_crit))**(1./3.)


def find_radius_mass(m_r, delta, z=0.0, cosmo=None):
    """
    Given a mass profile and an overdensity, find the radius 
    and mass (e.g. M200, r200)

    Parameters
    ----------
    m_r : RadialProfile
        The mass profile.
    delta : float
        The overdensity to compute the mass and radius for.
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    cosmo : yt ``Cosmology`` object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will 
        be used.
    """
    from yt.utilities.cosmology import Cosmology
    from scipy.optimize import bisect
    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    f = lambda r: 3.0*m_r(r)/(4.*np.pi*r**3) - delta*rho_crit
    r_delta = bisect(f, 0.01, 10000.0)
    return r_delta, m_r(r_delta)

if __name__ == '__main__':
    u = power_law_profile(1,2,3)
    u.to_binary("test.rp")
    print(u(5000))
    del u

    u = RadialProfile.from_binary("test.rp")
    print(u)
    print(u(5000))