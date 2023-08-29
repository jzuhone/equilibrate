-----------
Collections
-----------

To facilitate rapid implementation of various reasonable cluster scenarios, the ``cluster_generator`` package includes
several "collections" composed of best fit profiles to various real-world clusters. In this guide, we will discuss how
to use these collections and also how to add custom collections and help us to develop yet more of these collections.

Developer's Guide
-----------------

We've finishing covering the basics of actually accessing the built-in collection objects, but what about adding collections?

Every collection is housed in a custom class bearing its name. For example, the Vikhlinin 2006 fits are housed in the
:py:class:`collections.Vikhlinin06` class. All of these custom classes inherit from one common class; the :py:class:`collections.ClusterCollection` class.

The :py:class:`collections.ClusterCollection` class is largely just a wrapper for IO interaction. It is initialized with a single argument: ``path``,
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