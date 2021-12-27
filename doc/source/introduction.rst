.. _introduction:

Introduction
------------

Galaxy clusters are the largest gravitationally bound objects in the universe
and the current endpoints of the process of cosmological structure formation. 

Units
=====

The unit system assumed in ``cluster_generator`` is designed to use units
appropriate for cluster scales: 

* Length: :math:`{\rm kpc}`
* Time: :math:`{\rm Myr}`
* Mass: :math:`{\rm M_\odot}`

From these three, the units for other quantities, such as density, pressure, 
specific energy, gravitational potential, etc. are straighforwardly derived.  
These are the units which will be assumed for all inputs (whether arrays or 
scalars) to the functions in ``cluster_generator``, unless otherwise specified 
in the documentation and/or docstrings. What this means in practice is that if 
one supplies an input without units, it is assumed to be in the above units
depending on the type of input (e.g., position, velocity, etc.). If you supply
an input with units attached from `yt <https://yt-project.org>`_ or 
`unyt <http://unyt.readthedocs.org>`_, it will be converted to the above units 
internally before performing any calculations.

Some examples:

Most quantities which are returned as outputs of ``cluster_generator`` functions
have units attached, using `unyt <http://unyt.readthedocs.org>`_. These are usually in the above unit system.
For some output quantities, these units are sometimes used:

* Number density: :math:`{\rm cm^{-3}}`
* Temperature: :math:`{\rm keV}`
* Entropy: :math:`{\rm keV~cm^2}`