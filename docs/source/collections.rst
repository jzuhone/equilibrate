-----------
Collections
-----------

To help facilitate productive science, the :py:mod:`cluster_generator` library provides sets of documented galaxy clusters and
their fits to known profiles. These profiles are stored in a class called :py:class:`collection.ClusterCollection`, which allows the user
to interact with the stored profile data to generate known clusters directly from their observed behaviors.

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">
Available Datasets
------------------
.. card-carousel:: 2

    .. card:: :py:class:`collection.Vikhlinin06`


        **Methodology**: CHANDRA X-ray Observations
        ^^^
        Phenomenological models of the Vikhlinin et. al. 2006 clusters using Hernquist profiles for the total mass and a 5 parameter fitting procedure.
        +++
        | **Publication**: `Vikhlinin, A. et. al. 2006ApJ...640..691V <https://ui.adsabs.harvard.edu/abs/2006ApJ...640..691V/abstract>`_

    .. card:: :py:class:`collection.Ascasibar07`


        **Methodology**: CHANDRA X-ray Observations
        ^^^
        Gas temperature and gas density profiles for a set of 13 low-redshift, relaxed clusters with temperatures between 0.7-9 keV.
        +++
        | **Publication**: `Ascasibar, Y.; Diego, J. M. 2008MNRAS.383..369A <https://ui.adsabs.harvard.edu/abs/2008MNRAS.383..369A/abstract>`_

    .. card:: :py:class:`collection.Sanderson10`


        **Methodology**: CHANDRA X-ray Observations
        ^^^
        Gas temperature and gas density profiles for a set of 20 clusters from CHANDRA.
        +++
        | **Publication**: `Sanderson, A.J.R.; Ponman, T.J.  <https://ui.adsabs.harvard.edu/abs/2010MNRAS.402...65S/abstract>`_

.. raw:: html

   <hr style="height:2px;background-color:black">


Accessing Collections
---------------------

Built-In Collections
++++++++++++++++++++

For most purposes, built-in collections are the easiest way to go. There are several available in the library, the complete list can be seen in the :py:mod:`collection` documentation.
To load a built-in collection, one simply needs to call the ``.load`` method on the corresponding class. For example

.. code-block:: python

    >>> u = Vikhlinin06.load()

This will then provide you with access to the fully realized class. You can see the available clusters using the :py:attr:`collection.ClusterCollection.names` attribute.

.. code-block:: python

    >>> print(u.names)
    ... ['A133', 'A262', 'A383', 'A478', 'A907', 'A1413', 'A1795', 'A1991', 'A2029', 'A2390', 'RX J1159+5531', 'MKW 4', 'USGC S152']

If you want to load one of these as a :py:class:`model.ClusterModel` instance, simply use the following:

.. code-block:: python

    >>> model = u.load_model("A133")
    ... cluster_generator : [INFO     ] 2023-09-11 08:40:53,510 Loaded Vikhlinin et. al. 2006.
    ... cluster_generator : [INFO     ] 2023-09-11 08:40:53,519 Constructing ClusterModel. Method='from_dens_and_temp', gravity=Newtonian.
    ... 		[from_dens_and_temp]: Mon Sep 11 08:40:53 2023 Computing r, rho_g, T from profiles...[DONE]
    ... 		[from_dens_and_temp]: Mon Sep 11 08:40:53 2023 Computing calculating the pressure...[DONE]
    ... 		[from_dens_and_temp]: Mon Sep 11 08:40:53 2023 Computing the field...[DONE]
    ... 		[from_dens_and_temp]: Mon Sep 11 08:40:53 2023 Computing the mass and density fields...[DONE]
    ... cluster_generator : [WARNING  ] 2023-09-11 08:40:53,917 The model being generated has non-physical attributes.
    ... 		[from_dens_and_temp]: Mon Sep 11 08:40:53 2023 Passing to `from_scratch`...
    ... 		[from_scratch]: Mon Sep 11 08:40:53 2023 Checking for missing mass / density fields...[DONE]
    ... 		[from_scratch]: Mon Sep 11 08:40:53 2023 Determining the halo component...[DONE]
    ... 		[from_scratch]: Mon Sep 11 08:40:53 2023 Computing additional fields...[DONE]
    ... 		[from_scratch]: Mon Sep 11 08:40:53 2023 Initializing the ClusterModel...
    ... cluster_generator : [INFO     ] 2023-09-11 08:40:53,923 ClusterModel [ClusterModel object; gravity=Newtonian] has no virialization method. Setting to default = eddington
    ... cluster_generator : [INFO     ] 2023-09-11 08:40:53,923 Computing gravitational potential of ClusterModel object; gravity=Newtonian. gravity=Newtonian.
    ... ✔ cluster_generator : [INFO     ] Mon Sep 11 08:40:55 2023 Computed potential.

And there you go, you've got you :py:class:`model.ClusterModel` instance ready to go!

Custom Collections
++++++++++++++++++

If you're looking to make your own collections, the process is a bit more intricate, but still manageable.


Every collection is housed in a custom class bearing its name. For example, the Vikhlinin 2006 fits are housed in the
:py:class:`collections.Vikhlinin06` class. All of these custom classes inherit from one common class; the :py:class:`collection.ClusterCollection` class.

The :py:class:`collection.ClusterCollection` class is largely just a wrapper for IO interaction. It is initialized with a single argument: ``path``,
which is the path to the ``.yaml`` file containing the actual collection data. All that the :py:class:`collections.ClusterCollection` instance does
is read in the data contained in the ``.yaml`` for user interaction.

Let's begin by looking at a basic template for a collection:

.. code-block:: yaml

    #-----------------------------------------------------------#
    # =========== Basic Template for Collection =============== #
    #-----------------------------------------------------------#
    global:
      #------------------------------------------------------------------------#
      # The global dictionary of the collection contains collection wide data  #
      # regarding the name of the collection, the profiles to use, etc.        #
      #------------------------------------------------------------------------#
      name: "Sample Collection - 2023" # The name you give your collection

      profiles:
        #--------------------------------------------------------------------------------------#
        # The profiles dictionary should contain all of the profiles that the various          #
        # data objects are providing. There are two formats for profiles, either referencing   #
        # built in profiles or user defined profiles                                           #
        #--------------------------------------------------------------------------------------#
        first_profile: # these should be the field names.

          name: "a builtin profile" # This is the name that is used to look up the function. It must match.

          is_custom: false # If true, function must be an expression.

          parameters:
            #-------------------------------------------------#
            # These are the input parameters and their units  #
            #-------------------------------------------------#
            rho0: "Msun/kpc**3"
            r_c: "kpc"
            r_s: "kpc"
            alpha: ""
            beta: ""
            epsilon: ""
            gamma: ""

          function: null

        second_profile:

          name: "a custom profile"

          is_custom: true #--> This means we have to provide a function

          parameters:

            T0: "keV"
            a: ""
            b: ""
            c: ""
            r_t: "kpc"
            T_min: "keV"
            r_cool: "kpc"
            a_cool: ""

          # The function should be written as a multiline string and should be readable by the exec function in
          # python. It takes a parameter x and an argument list p, which should match the parameters above.
          function: |
            lambda x,p: p[0]*p[1]*x

      description: |
        This dataset contains fits to CHANDRA data from the paper by Vikhlinin et. al. 2006: 2006ApJ...640..691V.

      # The load_method is the method that is used to load the cluster models
      load_method: "from_dens_and_temp"

    objects:
      #-----------------------------------------------------------------------------------------------------#
      # This is where you place the actual objects. There are two ways to do this and they can be combined  #
      # -------                                                                                             #
      # 1. You can include the keys "uses" and "path" to load from a csv file, and                          #
      # 2. You can include custom objects as well                                                           #
      #-----------------------------------------------------------------------------------------------------#
      uses: "load_from_file"
      path: "./collections/Vikhlinin06.csv"

      extra_cluster:
        name: "ABELL Not a real cluster"
        desc: "some cute description"
        params:
          #------------------------------------------------------------------------------------------------#
          # Here you can include the parameters for each of the different profiles. Use the profile key    #
          # as the key here and use each of the parameter names followed by the correct value.             #
          #------------------------------------------------------------------------------------------------#