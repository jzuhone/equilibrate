.. _cluster_generator:
Cluster Generator
=================

+-------------------+----------------------------------------------------------+
| **Code**          + |black| |isort| |yt-project| |Pre-Commit|                +
+-------------------+----------------------------------------------------------+
| **Documentation** + |docformatter| |NUMPSTYLE| |docs|                        +
+-------------------+----------------------------------------------------------+
| **GitHub**        +  |CONTRIBUTORS| |COMMIT|                                 +
+-------------------+----------------------------------------------------------+
| **PyPi**          +                                                          +
+-------------------+----------------------------------------------------------+

.. raw:: html

   <hr style="height:10px;background-color:black">

Cluster Generator (CG) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CG provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, CG is intended to interface with
a number of N-body / hydrodynamics codes used in studies of galaxy clusters, reducing the headache of converting initial conditions between formats for different simulation softwares. CG's goal is to provide
comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

This repository contains the core package, which is constructed modularly to facilitate easy development by users to meet particular scientific use cases. All of the
necessary tools to get started building initial conditions are provided.

You can access the documentation :ref:`here <cluster_generator>`, or build it from scratch using the ``./docs`` directory in this source distribution.

Development occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on
the issues page of the repository.

For installation directions, visit the :ref:`installation page <installation>`.


.. raw:: html

   <hr style="color:black">
Features
========

.. grid:: 2

    .. grid-item::

        .. dropdown:: Radial Profiles

            Each of the following radial profiles is included for use in cluster construction:

            .. dropdown:: NFW Profiles

              - :py:func:`~radial_profiles.nfw_density_profile`
              - :py:func:`~radial_profiles.nfw_mass_profile`
              - :py:func:`~radial_profiles.snfw_density_profile`
              - :py:func:`~radial_profiles.snfw_mass_profile`
              - :py:func:`~radial_profiles.cored_snfw_density_profile`
              - :py:func:`~radial_profiles.cored_snfw_mass_profile`
              - :py:func:`~radial_profiles.tnfw_density_profile`
              - :py:func:`~radial_profiles.tnfw_mass_profile`

            .. dropdown:: Hernquist Profiles

              - :py:func:`~radial_profiles.hernquist_density_profile`
              - :py:func:`~radial_profiles.hernquist_mass_profile`
              - :py:func:`~radial_profiles.cored_hernquist_density_profile`

            .. dropdown:: Einasto Profiles

              - :py:func:`~radial_profiles.einasto_density_profile`
              - :py:func:`~radial_profiles.einasto_mass_profile`

            .. dropdown:: General Profiles

              - :py:func:`~radial_profiles.power_law_profile`
              - :py:func:`~radial_profiles.constant_profile`
              - :py:func:`~radial_profiles.beta_model_profile`

            .. dropdown:: Paper Specific Profiles

              - **Vikhlinin et. al. 2006**
                - :py:func:`~radial_profiles.vikhlinin_density_profile`
                - :py:func:`~radial_profiles.vikhlinin_temperature_profile`
              - **Ascasibar & Markevitch 2006**
                - :py:func:`~radial_profiles.am06_density_profile`
                - :py:func:`~radial_profiles.am06_temperature_profile`

            .. dropdown:: Entropy Profiles

              - :py:func:`~radial_profiles.walker_entropy_profile`
              - :py:func:`~radial_profiles.baseline_entropy_profile`
              - :py:func:`~radial_profiles.broken_entropy_profile`

    .. grid-item::

        .. dropdown:: Gravitational Theories

            .. note::

                We're actively developing non-Newtonian gravitational paradigms! Reach out to us
                if you have any questions about utilizing this feature before it's officially released.


    .. grid-item::

        .. dropdown:: Implemented Codes

            ``cluster_generator`` provides ready-to-use initial condition generation tools for **all** of the following
            codes:

            - RAMSES
            - ATHENA++
            - AREPO
            - GAMER
            - FLASH
            - GIZMO
            - ENZO

    .. grid-item::

        .. dropdown:: Available Datasets

            In order to facilitate a variety of different scientific use-cases for the CG code, we intend to
            include a database of known systems and their best fit models from the literature in a future release.


Resources
=========

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: _images/index/stopwatch_icon.png

        Quickstart Guide
        ^^^^^^^^^^^^^^^^
        New to ``cluster_generator``? The quickstart guide is the best place to start learning to use all of the
        tools that we have to offer!

        +++

        .. button-ref:: getting_started
            :expand:
            :color: secondary
            :click-parent:

            To The Quickstart Page

    .. grid-item-card::
        :img-top: _images/index/lightbulb.png

        Examples
        ^^^^^^^^
        Have some basic experience with ``cluster_generator``, but want to see a guide on how to execute a particular task? Need
        to find some code to copy and paste? The examples page contains a wide variety of use case examples and explainations
        for all of the various parts of the ``cluster_generator`` library.

        +++

        .. button-ref:: examples
            :expand:
            :color: secondary
            :click-parent:

            To the Examples Page

    .. grid-item-card::
        :img-top: _images/index/book.svg

        User References
        ^^^^^^^^^^^^^^^^
        The user guide contains comprehensive, text based explainations of the backbone components of the ``cluster_generator`` library.
        If you're looking for information on the underlying code or for more details on particular aspects of the API, this is your best resource.

        +++

        .. button-ref:: codes
            :expand:
            :color: secondary
            :click-parent:

            To the User Guide

    .. grid-item-card::
        :img-top: _images/index/api_icon.png

        API Reference
        ^^^^^^^^^^^^^

        Doing a deep dive into our code? Looking to contribute to development? The API reference is a comprehensive resource
        complete with source code and type hinting so that you can find every detail you might need.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            API Reference


Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   getting_started
   theory
   examples
   api
   development

Related Projects
================

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: _images/index/PyXSIM.png

        PyXSIM
        ^^^^^^

        Convert your ``cluster_generator`` systems into synthetic photon event lists for use in simulating observations using X-ray observatories using PyXSIM.
        ``cluster_generator`` is designed to easily interface with this library to provide as much ease as possible when building simulated observations of clusters.

        +++

        .. button-link:: http://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/
            :expand:
            :color: secondary
            :click-parent:

            Documentation


    .. grid-item-card::
        :img-top: _images/index/SOXS.png

        SOXS
        ^^^^

        Coupled with PyXSIM, SOXS is a instrument simulation tool for turning mock photon counts into realistic X-ray observations specific to the
        behavior of specific instruments like *CHANDRA*, *XMM-Newton*, and *NuSTAR*.

        +++

        .. button-link:: https://www.lynxobservatory.com/soxs
            :expand:
            :color: secondary
            :click-parent:

            Documentation





Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org
.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
.. |testing| image:: https://github.com/jzuhone/cluster_generator/actions/workflows/test.yml/badge.svg
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/
.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |CONTRIBUTORS| image:: https://img.shields.io/github/contributors/jzuhone/cluster_generator
    :target: https://github.com/jzuhone/cluster_generator/graphs/contributors
.. |COMMIT| image:: https://img.shields.io/github/last-commit/jzuhone/cluster_generator
.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9.svg
    :target: https://numpydoc.readthedocs.io/en/latest/format.html
.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg
    :target: https://github.com/PyCQA/docformatter
