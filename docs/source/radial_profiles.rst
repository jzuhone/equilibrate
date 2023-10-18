.. _radial_profiles:

Radial Profiles
---------------

.. raw:: html

   <hr style="height:10px;background-color:black">

Under the hood, the CGP relies on functions of the cluster's radius to represent most physical variables. These are then used
to carry out the mathematics necessary to produce the self consistent clusters necessary for simulation. The :py:mod:`radial_profiles` module
and its core class :py:class:`radial_profiles.RadialProfile` provide all of the necessary additional structure needed for these
radial profiles wrapped around standard python callables. On this page, you'll find information on the built-in radial profiles that
CGP provides, how to use them, and how to define your own custom :py:class:`radial_profiles.RadialProfile`.

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">


Built-In Radial Profiles
========================

The following radial profiles are built-in to the CGP framework. You can call them just as you would a standard function.
Click on the name of each profile to access more comprehensive information about it.

.. py:currentmodule:: radial_profiles

.. autosummary::

    ad07_density_profile
    ad07_temperature_profile
    am06_density_profile
    am06_temperature_profile
    baseline_entropy_profile
    beta_model_profile
    broken_entropy_profile
    constant_profile
    cored_hernquist_density_profile
    cored_snfw_density_profile
    cored_snfw_mass_profile
    cored_snfw_total_mass
    einasto_density_profile
    einasto_mass_profile
    hernquist_density_profile
    hernquist_mass_profile
    nfw_density_profile
    nfw_mass_profile
    power_law_profile
    snfw_conc
    snfw_density_profile
    snfw_mass_profile
    snfw_total_mass
    tnfw_density_profile
    tnfw_mass_profile
    vikhlinin_density_profile
    vikhlinin_temperature_profile
    walker_entropy_profile


Using Radial Profiles
=====================

:py:class:`radial_profiles.RadialProfile` instances work just like any other function; you can call them on arrays or on individual
values and get an output just as you usually would. For example,

.. code-block:: python

    from cluster_generator.radial_profiles import constant_profile

    p = constant_profile(10)
    print(p(6))
    >>> 10

On top of the expected behaviors as a function, :py:class:`radial_profiles.RadialProfile` have additional functionality which can be of great
use. The :py:meth:`radial_profiles.RadialProfile.cutoff` and :py:meth:`radial_profiles.RadialProfile.add_core` can be used to truncate a profile
at a certain radius or to add a core to the center of the profile respectively.

You can also save :py:class:`radial_profiles.RadialProfile` instances to disk using a binary serialization with :py:meth:`radial_profiles.RadialProfile.to_binary`. These can be read again
with the :py:meth:`radial_profiles.RadialProfile.from_binary`. Finally, there is the :py:meth:`radial_profiles.RadialProfile.built_in` which allows the user to look up
a built-in profile by string name instead of haviing to load the function in a priori.

Creating a Radial Profile
=========================
If you want to create your own radial profile object, the process is extremely easy. All you have to do is build the class from a function as
so

.. code-block:: python

    from cluster_generator.radial_profiles import constant_profile, RadialProfile

    new_profile = lambda x,a,b: x*a*b

    prof = RadialProfile(new_profile,name="My Random Profile")

If your goal is to contribute a radial profile to the code permanently, we request that you use the following template. All radial profiles
should be placed directly into the ``cluster_generator/radial_profiles.py`` file.

.. code-block:: python

    def your_function_here(x,a,b,*args):
        """
        Radial Profile from <insert citation here> representing ... (give explanation).

        Parameters
        ----------
        a: float
            Desc.
        b: float
            Desc

        < Complete the doc string >

        Return
        ------
        float
        References
        ----------
        .. [] --> Your reference here.
        """
        # Define your function here

        return "<The value of the function>"

Once you have created the custom function, you need only add it to the :py:class:`radial_profiles.RadialProfile` class's
``built_in`` profile with the appropriate details.
