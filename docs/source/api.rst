API
===

Below are all the various code components of ``cluster_generator``. Each class, method, function,
and variable in the codebase is documented under its relevant heading. If you're looking for a
basic introduction to the code, we suggest starting with the :ref:`getting_started` page.

Galaxy Cluster Modeling
-----------------------

This section contains modules related to the modeling of galaxy clusters.
These modules provide tools for generating initial conditions, defining radial profiles,
managing particle data, and building comprehensive cluster models.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    radial_profiles
    ics
    model
    particles

External Codes
--------------

Modules in this section interface with external codes and software packages.
They enable ``cluster_generator`` to integrate with third-party tools, ensuring
seamless interoperability and leveraging external computational capabilities. Most important of these is
frontend interactivity with simulation software of a variety of types.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    codes

Physics
-------

These modules provide a collection of physical relations, constants, and utilities specific to galaxy cluster physics.
This includes virial relations, equations governing cluster dynamics, and functions for calculating various cluster properties.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    relations
    virial
    fields

Utilities
---------

The utilities section contains general-purpose functions and classes that support various aspects of the ``cluster_generator``
framework. These include configuration management, logging, data handling, and other supportive tasks essential
for running simulations and models effectively.

.. autosummary::
    :toctree: _as_gen
    :template: module.rst
    :recursive:

    utils
