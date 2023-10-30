.. image:: _images/cluster_generator_logo.png


The Cluster Generator Project
=============================

|yt-project| |precom| |docs| |testing| |Github Page| |Pylint| |coverage| |ncodes|

.. raw:: html

   <hr style="height:10px;background-color:black">


`Cluster Generator <https://jzuhone.github.io/cluster_generator>`_ (CG) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CG provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, CG is intended to interface with
a number of N-body / hydrodynamics codes used in studies of galaxy clusters, reducing the headache of converting initial conditions between formats for different simulation softwares. GCP's goal is to provide

comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

This repository contains the core package, which is constructed modularly to facilitate easy development by users to meet particular scientific use cases. All of the
necessary tools to get started building initial conditions are provided.

You can access the documentation `here <http://jzuhone.github.io/cluster_generator>`_, or build it from scratch using the ``./docs`` directory in this source distribution.

Development occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on
the issues page of the repository.

For installation directions, visit the `getting started page <https://jzuhone.github.io/cluster_generator/build/html/Getting_Started.html>`_.


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
              - :py:func:`~radial_profiles.cored_hernquist_mass_profile`

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


            ``cluster_generator`` not only provides initial condition generation capacity, but also provides a

            comprehensive catalog of alternative gravity theories to explore. The following are built-in, but adding more
            is a relatively simple task:

            - :ref:`Newtonian Gravity <gravity>`

            .. dropdown:: MONDian Gravities

                - :ref:`AQUAL <aqual>`
                - :ref:`QUMOND <qumond>`


    .. grid-item::

        .. dropdown:: Implemented Codes

            The CGP provides end-to-end initial condition generation tools for **all** of the following
            codes:

            - :ref:`RAMSES <ramses>`
            - :ref:`ATHENA++ <athena>`
            - :ref:`AREPO <arepo>`
            - :ref:`GAMER <gamer>`
            - :ref:`FLASH <flash>`
            - :ref:`GIZMO <gizmo>`
            - :ref:`ENZO <enzo>`

    .. grid-item::

        .. dropdown:: Available Datasets

            The :ref:`Collections <collections>` system provides users access to pre-built galaxy clusters from the available literature. Cluster fits
            are available for all of the following papers:

            - `Vikhlinin et. al. 2006 <https://ui.adsabs.harvard.edu/abs/2006ApJ...640..691V/abstract>`_

    .. grid-item::

        .. dropdown:: Automated Non-Physicality Correction

            The CGP provides a purpose built algorithm for non-physical corrections in initialized clusters to reduce
            labor overhead in the generation of the initial conditions. For more information, visit the :ref:`correction` page.

Resources
=========

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: _images/index/stopwatch_icon.png

        Quickstart Guide
        ^^^^^^^^^^^^^^^^
        New to the CGP? The quickstart guide is the best place to start learning to use all of the
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
        Have some basic experience with the CGP, but want to see a guide on how to execute a particular task? Need
        to find some code to copy and paste? The examples page contains a wide variety of use case examples and explanations
        for all of the various parts of the CGP library.

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
        The user guide contains comprehensive, text based explanations of the backbone components of the CGP library.
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



Related Projects
================

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: _images/index/PyXSIM.png

        PyXSIM
        ^^^^^^

        Convert your CGP generated systems into synthetic photon event lists for use in simulating observations using X-ray observatories using PyXSIM.
        the CGP is designed to easily interface with this library to provide as much ease as possible when building simulated observations of clusters.

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

Reference Pages
===============

.. raw:: html

    <style>
    .ag-format-container {
      width: 900px;
      margin: 0 auto;
    }


    body {
    }
    .ag-courses_box {
      display: -webkit-box;
      display: -ms-flexbox;
      display: flex;
      -webkit-box-align: start;
      -ms-flex-align: start;
      align-items: flex-start;
      -ms-flex-wrap: wrap;
      flex-wrap: wrap;
      padding: 50px 0;
    }
    .ag-courses_item {
      -ms-flex-preferred-size: calc(33.33333% - 30px);
      flex-basis: calc(33.33333% - 30px);
      margin: 0 15px 30px;

      overflow: hidden;
      border-radius: 10px;
    }
    .ag-courses-item_link {
      background-color: #FFF;
      display: block;
      border: 5px black solid;
      padding: 30px 20px;

      overflow: hidden;

      position: relative;
    }
    .ag-courses-item_link:hover,
    .ag-courses-item_link:hover .ag-courses-item_date {
      text-decoration: none;
      color: #000;
    }
    .ag-courses-item_link:hover .ag-courses-item_bg {
      -webkit-transform: scale(10);
      -ms-transform: scale(10);
      transform: scale(10);
    }
    .ag-courses-item_title {
      min-height: 87px;
      margin: 0 0 25px;

      overflow: hidden;

      font-weight: bold;
      font-size: 30px;
      color: #000;

      z-index: 2;
      position: relative;
    }
    .ag-courses-item_date-box {
      font-size: 18px;
      color: #000;

      z-index: 2;
      position: relative;
    }
    .ag-courses-item_date {
      font-weight: bold;
      color: #66a4e4;

      -webkit-transition: color .5s ease;
      -o-transition: color .5s ease;
      transition: color .5s ease
    }
    .ag-courses-item_bg {
      height: 128px;
      width: 128px;
      background-color: #66a4e4;

      z-index: 1;
      position: absolute;
      top: -75px;
      right: -75px;

      border-radius: 50%;

      -webkit-transition: all .5s ease;
      -o-transition: all .5s ease;
      transition: all .5s ease;
    }
    .ag-courses_item:nth-child(2n) .ag-courses-item_bg {
      background-color: #6dd162;
    }
    .ag-courses_item:nth-child(3n) .ag-courses-item_bg {
      background-color: #59aa4c;
    }
    .ag-courses_item:nth-child(4n) .ag-courses-item_bg {
      background-color: #75ce8c;
    }
    .ag-courses_item:nth-child(5n) .ag-courses-item_bg {
      background-color: #d179e3;
    }
    .ag-courses_item:nth-child(6n) .ag-courses-item_bg {
      background-color: #e37dc1;
    }
    .ag-courses_item:nth-child(7n) .ag-courses-item_bg {
      background-color: #75e6c3;
    }
    .ag-courses_item:nth-child(8n) .ag-courses-item_bg {
      background-color: #4fd1d9;
    }
    .ag-courses_item:nth-child(8n) .ag-courses-item_bg {
      background-color: #a993e3
    }
    .ag-courses_item:nth-child(8n) .ag-courses-item_bg {
      background-color: #48b8e2;
    }




    @media only screen and (max-width: 979px) {
      .ag-courses_item {
        -ms-flex-preferred-size: calc(50% - 30px);
        flex-basis: calc(50% - 30px);
      }
      .ag-courses-item_title {
        font-size: 24px;
      }
    }

    @media only screen and (max-width: 767px) {
      .ag-format-container {
        width: 96%;
      }

    }
    @media only screen and (max-width: 639px) {
      .ag-courses_item {
        -ms-flex-preferred-size: 100%;
        flex-basis: 100%;
      }
      .ag-courses-item_title {
        min-height: 72px;
        line-height: 1;

        font-size: 24px;
      }
      .ag-courses-item_link {
        padding: 22px 40px;
      }
      .ag-courses-item_date-box {
        font-size: 16px;
      }
    }
    </style>
    <div class="ag-format-container">
      <div class="ag-courses_box">
        <div class="ag-courses_item">
          <a href="models.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Cluster Models
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Beginner
              </span>
            </div>
          </a>
        </div>


        <div class="ag-courses_item">
          <a href="virialization.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Virialization
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Intermediate
              </span>
            </div>
          </a>
        </div>

        <div class="ag-courses_item">
          <a href="correction.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Non-Physical Corrections
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Intermediate
              </span>
            </div>
          </a>
        </div>

        <div class="ag-courses_item">
          <a href="radial_profiles.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Radial Profiles
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Beginner
              </span>
            </div>
          </a>
        </div>
        <div class="ag-courses_item">
          <a href="fields.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Fields
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Advanced
              </span>
            </div>
          </a>
        </div>
        <div class="ag-courses_item">
          <a href="examples.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Examples
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Beginner
              </span>
            </div>
          </a>
        </div>
        <div class="ag-courses_item">
          <a href="codes.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Codes
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Beginner
              </span>
            </div>
          </a>
        </div>
        <div class="ag-courses_item">
          <a href="initial_conditions.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Initial Conditions
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Beginner
              </span>
            </div>
          </a>
        </div>
        <div class="ag-courses_item">
          <a href="particles.html" class="ag-courses-item_link">
            <div class="ag-courses-item_bg"></div>

            <div class="ag-courses-item_title">
              Particles
            </div>

            <div class="ag-courses-item_date-box">
              Level:
              <span class="ag-courses-item_date">
                Beginner
              </span>
            </div>
          </a>
        </div>
      </div>
    </div>


Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Additional Pages
================

.. toctree::
   :maxdepth: 1

   getting_started
   examples
   models
   codes
   api

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://jzuhone.github.io/cluster_generator/build/html/index.html
.. |precom| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit

.. |testing| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg
.. |Pylint| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg
.. |Github Page| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg
.. |coverage| image:: https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg
   :target: https://coveralls.io/github/Eliza-Diggins/cluster_generator
.. |ncodes| image:: https://img.shields.io/static/v1?label="Implemented%20Sim.%20Codes"&message="7"&color="red"
    :target: https://jzuhone.github.io/cluster_generator/build/html/codes.html
