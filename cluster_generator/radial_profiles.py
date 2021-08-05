import numpy as np


_nfw_factor = lambda conc: 1.0/(np.log(conc+1.0)-conc/(1.0+conc))


class RadialProfile:
    def __init__(self, profile):
        if isinstance(profile, RadialProfile):
            self.profile = profile.profile
        else:
            self.profile = profile

    def __call__(self, r):
        return self.profile(r)

    def _do_op(self, other, op):
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

    def add_core(self, r_core, alpha):
        """
        Add a small core with radius *r_core* to the profile by
        multiplying it by 1-exp(-(r/r_core)**alpha).

        Parameters
        ----------
        r_core : float 
            The core radius in kpc. 
        """
        def _core(r):
            x = r/r_core
            ret = 1.0-np.exp(-x**alpha)
            return self.profile(r)*ret
        return RadialProfile(_core)

    @classmethod
    def from_array(cls, r, f_r):
        """
        Generate a callable radial profile using an array of radii
        and an array of values. 

        Parameters
        ----------
        r : NumPy array
            Array of radii in kpc.
        f_r : NumPy array
            Array of profile values in the appropriate units.
        """
        from scipy.interpolate import InterpolatedUnivariateSpline
        f = InterpolatedUnivariateSpline(r, f_r)
        return cls(f)

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


def beta_model_profile(rho_c, r_c, beta):
    """
    A beta-model density profile (Cavaliere A., 
    Fusco-Femiano R., 1976, A&A, 49, 137).

    Parameters
    ----------
    rho_c : float
        The core density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    beta : float
        The beta parameter.
    """
    p = lambda r: rho_c*((1+(r/r_c)**2)**(-1.5*beta))
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
    """
    if gamma is None:
        gamma = 3.0
    profile = lambda r: rho_0*(r/r_c)**(-0.5*alpha) * \
        (1.+(r/r_c)**2)**(-1.5*beta+0.25*alpha) * \
        (1.+(r/r_s)**gamma)**(-0.5*epsilon/gamma)
    return RadialProfile(profile)


def hernquist_density_profile(M_0, a):
    """
    A Hernquist density profile (Hernquist, L. 1990,
    ApJ, 356, 359).

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
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
    """
    def _nfw(r):
        x = r/r_s
        return 4*np.pi*rho_s*r_s**3*(np.log(1+x)-x/(1+x))
    return RadialProfile(_nfw)


def tnfw_density_profile(rho_s, r_s, r_t):
    """
    A truncated NFW density profile ().

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.
    """
    def _tnfw(r):
        profile = rho_s/((r/r_s)*(1+r/r_s)**2)
        profile /= (1+(r/r_t)**2)
        return profile
    return RadialProfile(_tnfw)


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
    """
    from yt.utilities.cosmology import Cosmology
    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    rho_s = delta*rho_crit*conc**3*_nfw_factor(conc)/3.
    return rho_s


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
    """
    def _snfw(r):
        x = r/a
        return M*(1.-(2.+3.*x)/(2.*(1.+x)**1.5))
    return RadialProfile(_snfw)


def snfw_total_mass(Mr, r, a):
    mp = snfw_mass_profile(1.0, a)
    return Mr/mp(r)


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
    """
    return 0.76*conc_nfw+1.36


def cored_snfw_total_mass(Mr, r, a, r_c):
    mp = cored_snfw_mass_profile(1.0, a, r_c)
    return Mr/mp(r)


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
    """
    alpha = -1.-n*(c-1.)/(c-a/a_c)
    beta = 1.-n*(1.-a/a_c)/(c-a/a_c)
    p = lambda r: rho_0*(1.+r/a_c)*(1.+r/a_c/c)**alpha*(1.+r/a)**beta
    return RadialProfile(p)


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
        The mass of the object in Msun.
    radius : float
        The radius that the ``mass`` corresponds to in kpc.

    Examples
    --------
    >>> rho_0 = 1.0
    >>> a = 600.0
    >>> a_c = 60.0
    >>> c = 0.17
    >>> alpha = -2.0
    >>> beta = -3.0
    >>> gas_density = am06_density_profile(rho_0, a, a_c, c, alpha, beta)
    >>> M200 = 1.0e14
    >>> r200 = 900.0
    >>> gas_density = rescale_profile_by_mass(gas_density, M0, r200)
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
