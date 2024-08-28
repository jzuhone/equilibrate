.. _development:
============
Development
============

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

Cluster Generator Overview
--------------------------

At its core, CG is a tool for constructing idealized models of galaxy clusters which can (in turn) be applied in a variety
of scientific applications. Depending on the particular needs of the user, the complexity of that "idealized model" may vary
significantly, as may the fundamental principles applied in the creation of the model. Because of this, continued development of
this library to expand its applicability can begin to interfere with its ease-of-use. As such, when participating in the development of
CG, we encourage developers to design their code in such a way as to integrate well with the following general structure:

- User determined choices regarding fundamental physics should be implemented as classes in a relevant module. For example:

  - Different gravitational theories should be subclasses or instances of some gravity class in an independent module.
  - If you're adding compatibility for different equations of state, these should subclass from a base class and have their own module.
  - Different virial equilibrium approaches (LMA vs. Eddington) should be represented by different classes within the :py:mod:`virial` module.

- The core class in this library is the :py:class:`model.ClusterModel` class, which represents a **complete** model of a given galaxy cluster.

  - Additions to the code which increase the complexity of, or introduce a new kind of, model should always be implemented in such a way
    as to subclass from :py:class:`model.ClusterModel`. This may include making changes to the superclass; however, API changes to existing
    model subclasses should be minimized.

  - Models should be self consistent and complete. The underlying physics from which a model was generated should be retained by the model itself.
    In cases where a particular gravitational paradigm, equation of state, or some other (fundamental) physics was altered, the model should retain
    some record of that underlying physics.

  - Models only ever contain a single component. Multi-component systems should be implemented as :py:class:`ics.ClusterICs`.

- Models of compatible types should be arbitrarily combine-able as :py:class:`ics.ClusterICs`.

- Simulation codes should also have a self-consistent frontend. Frontends are complete and either compatible with all model types or
  raise informative errors regarding models that are not compatible.


Library Structure
-----------------

Utilities
'''''''''

Code should be added to the :py:mod:`utilities` module only if its purpose extends beyond the scope of the module being developed. For example,
a mathematical function with general applications should, even if it is only used in a particular module, still be placed in :py:mod:`utilities` in anticipation
of its potential use elsewhere. This helps to prevent circuitous importing conventions and ensures that code duplication is minimized.

Additionally, functionality involving the logging system, the configuration, or type checking should be entirely contained in the corresponding
utility module.

.. hint::

    One instance in which this may frequently occur is type hinting. If a specific alias is used in the module under development
    for a common type configuration, it should be added to the :py:mod:`utilities.types` module, not to the physics module under development.


Model Building Blocks
''''''''''''''''''''''

Separate aspects of the sub-model physics should occupy separate modules. For example:

- :py:mod:`radial_profiles` for different initial profiles.
- :py:mod:`virial` for different virial equilibrium models.

When developing parts of CG, modules should correspond to the most general physics to which they apply.

.. admonition:: example

    Consider the case in which the developer is implementing ellipsoidal profiles for use in non-spherical galaxy cluster models.
    The structure of such a change would need to be generally as follows:

    - :py:mod:`model` generalized so that :py:class:`model.ClusterModel` can be non-spherical. A new ``SphericalClusterModel`` class defined
      as well as the new ``EllipsoidalClusterModel`` class.
    - :py:mod:`radial_profiles` renamed to something like ``profiles`` and a more general ``Profile`` class created from which
      :py:class:`radial_profiles.RadialProfile` might be subclassed.

    In this way, the :py:mod:`radial_profiles` module is expanded to the most general context and the namespace is not littered with
    similar sounding module names.

Models
''''''

As with all parts of CG, models should always subclasses of :py:class:`model.ClusterModel`, even if doing so requires that
:py:class:`model.ClusterModel` be generalized. All model classes should be **self-consistent**, specifically in that they should be
writable and readable from disk. At a minimum, models should have an HDF5 representation with a similar API to that of the superclass.

- The physical properties of models are fields; all model classes should have a fields attribute which functions in the same manner as
  :py:class:`model.ClusterModel`'s :py:attr:`~model.ClusterModel.fields`.
- Models should be convertible into a :py:class:`particles.ClusterParticles`-like class.
- Models can be combined to form :py:class:`ics.ClusterICs` or some descendant thereof.

Initial Conditions and Code Frontends
'''''''''''''''''''''''''''''''''''''

.. note::

    Guidance regarding developing code frontends will be added in a future version of the documentation.

External Packages
'''''''''''''''''

Compatibility with external software allows cluster generator to maximize its utility; however, the following should be considered
when adding an interface with another piece of software.

- **Minimize dependencies**: If it's possible to create the interface without requiring new dependencies, do so. This reduces the
  potential for bugs to emerge as both projects continue to develop.
- **Let Users Do the Work**: If CG's models can be used externally, don't add hard-coded implementations to the library. Assume
  that users can determine how to use the external software and provide only the necessary infrastructure to move models from CG to
  the third party software.
