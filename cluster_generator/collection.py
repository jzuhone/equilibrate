"""
Implemented collections of cluster models.
"""
import os
import pathlib as pt

import pandas as pd
import yaml

import cluster_generator.radial_profiles as rprofs
from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog

# -------------------------------------------------------------------------------------------------------------------- #
# Setup ============================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #

# -- finding the base storage directory -- #
collections_directory = os.path.join(pt.Path(__file__).parents[0], "bin", "collections")


class ClusterCollection:
    """
    Base class representation of cluster collections.

    """
    _required_global_attributes = ["name", "description", "load_method"]

    #  Dunder Methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self, path):
        """
        Initializes the :py:class:`collection.ClusterCollection` instance.

        Parameters
        ----------
        path: str
            The path to the ``.yaml`` file containing the underlying data.

        Raises
        ------
        yaml.YAMLError
            Occurs if there is an error in the underlying yaml formatting.
        FileNotFoundError
            Raised if :py:attr:`~collection.ClusterCollection.path` fails to actually point to a real file.
        CollectionsError
            Raised if there are missing parts of the collection data structure.
        """
        #  Setup
        # ------------------------------------------------------------------------------------------------------------ #
        _fail_with_alert = False  # Tells the loader if something went wrong to alert or not.
        #: The path to the underlying ``.yaml`` file.
        self.path = path
        #: The name of the dataset. This is obtained from the data file.
        self.name = None
        #: The description of the dataset. This is obtained from the data file.
        self.description = None
        #: The method used to load the clusters. Obtained from the data file.
        self.load_method = None

        #  Loading the YAML file
        # ------------------------------------------------------------------------------------------------------------ #
        try:
            with open(path, "r+") as yf:
                _col_data = yaml.load(yf, yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"There is no {os.path.join(os.getcwd(), path)}. Is there a typo?")
        except yaml.YAMLError as yex:
            raise yaml.YAMLError(
                f"The yaml file at {os.path.join(os.getcwd(), path)} failed to load. message = {yex.__repr__()}")

        #  Managing global / meta data
        # ------------------------------------------------------------------------------------------------------------ #
        if "global" not in _col_data:
            raise CollectionsError(f"Failed to locate the 'global' key in {path}.", section="global")
        for attr in self._required_global_attributes:
            try:
                setattr(self, attr, _col_data["global"][attr])
            except KeyError:
                raise CollectionsError(f"Failed to locate `{attr}` key in globals for {path}.",
                                       section=f"global.{attr}")
        #  Managing profiles
        # ------------------------------------------------------------------------------------------------------------ #
        # -- making sure they exist -- #
        if "profiles" not in _col_data["global"]:
            raise CollectionsError(f"Failed to find profiles in {path}", section=f"global.profiles")

        self._parameters, self._profiles = {}, {}
        for profile, data in _col_data["global"]["profiles"].items():
            # -- sanity checks -- #
            if not data["is_custom"]:  # This isn't custom, make sure there's a name
                if "name" not in data:
                    mylog.warning(f"The profile {profile} is not custom, but has no name. Skipping.")
                    _fail_with_alert = True
                    continue
            else:
                if "function" not in data:
                    mylog.warning(f"The profile {profile} is custom but has no function. Skipping.")
                    _fail_with_alert = True
                    continue

            # -- actually loading -- #
            if not data["is_custom"]:
                # this is not a custom profile. We just check it exists.
                if not hasattr(rprofs, data['name']):
                    mylog.warning(
                        f"The profile {profile} doesn't correspond to any radial profile. (name = {data['name']}).")
                    _fail_with_alert = True
                    continue
                else:
                    self.profiles[profile] = getattr(rprofs, data['name'])
                    self.parameters[profile] = data["parameters"]
            else:
                # this is a custom module, we just if function exists
                if not "function" in data:
                    mylog.warning(f"The profile {profile} doesn't have a function.")
                    _fail_with_alert = True
                    continue
                else:
                    try:
                        self.profiles[profile] = exec(data["function"])
                    except Exception as ex:
                        mylog.warning(f"The profile {profile} didn't execute correctly. Error = {ex.__repr__()}")
                        _fail_with_alert = True
                        continue
                    self.parameters[profile] = data["parameters"]

        #  Loading the actual datasets
        # -------------------------------------------------------------------------------------------------------- #
        self.objs = {}
        if "uses" in _col_data["objects"]:
            # -- We are going to be loading via external approach -- #
            if _col_data["objects"]["uses"] == "load_from_file":
                del _col_data["objects"]["uses"]
                # loading in the necessary start information #
                try:
                    assert "path" in _col_data["objects"]
                    df = pd.read_csv(os.path.join(pt.Path(self.path).parents[0], _col_data["objects"]["path"]))
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    df.columns = df.columns.str.rstrip()
                    del _col_data['objects']['path']
                except FileNotFoundError:
                    mylog.warning(
                        f"The data file {os.path.join(os.getcwd(), _col_data['objects']['path'])} doesn't exist. Skipping.")
                    _fail_with_alert = True
                    del _col_data['objects']['path']
                except AssertionError:
                    mylog.warning(f"No path for external loading was specified.")
                    _fail_with_alert = True
                # data manipulation
                self.objs = {
                    **self.objs,
                    **{df.iloc[i, 0]: {u: v for u, v in zip(df.columns[0:], df.iloc[i, 0:])} for i in
                       range(len(df))}
                }
            else:
                mylog.warning(
                    f"The uses option {_col_data['objects']['uses']} is not valid")
                _fail_with_alert = True
        else:
            pass

        # -- Standard Loading -- #
        for k, v in _col_data["objects"].items():
            self.objs[k] = v

        if _fail_with_alert:
            mylog.warning(f"Loaded {self.name} with warnings.")
        else:
            mylog.info(f"Loaded {self.name}.")

    def __repr__(self):
        return f"<ClusterCollection object> with (N={len(self.objs)},name={self.name})."

    def __str__(self):
        return f"ClusterCollection; name = {self.name}, length = {len(self.objs)}."

    def __contains__(self, item):
        return item in self.objs

    def __getitem__(self, item):
        return self.objs[item]

    def __setitem__(self, key, value):
        self.objs[key] = value

    #  Properties
    # ---------------------------------------------------------------------------------------------------------------- #
    @property
    def profiles(self):
        """
        The :py:class:`radial_profiles.RadialProfile` instances corresponding to the fitting functions of the dataset. The keys of this
        dictionary are the **field names** corresponding to the relevant profile.
        """
        return self._profiles

    @property
    def parameters(self):
        """
        Dictionary of parameters from the underlying dataset for each of the profiles in :py:attr:`collection.ClusterCollection.profiles`.
        """
        return self._parameters

    @property
    def names(self):
        """
        The names of the clusters available in the dataset.
        """
        return list(self.objs.keys())

    #  Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    def load_model(self, model_name, r_min, r_max, num_points=1000, gravity="Newtonian", **kwargs):
        """
        Generates a :py:class:`model.ClusterModel` instance representation of one of the galaxy clusters in the dataset specified
        by `model_name`.

        Parameters
        ----------
        model_name: str
            The name of the galaxy cluster to load.
        r_min: float
            The minimum radius at which to sample the dataset.
        r_max: float
            The maximum radius at which to sample the dataset.
        num_points: int
            The number of sample points for the profiles.
        gravity: str
            The gravity theory to use.
        **kwargs: optional
            additional parameters to pass through the loading system.

        Returns
        -------
        :py:class:`~model.ClusterModel`
            The finished cluster model.
        """
        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if model_name not in self:
            raise ValueError(
                f"The model {model_name} does not correspond to any of the data entries.\n  Options are: {self.names}")

        #  Cleanup
        # ------------------------------------------------------------------------------------------------------------ #
        # - grabbing the profiles - #
        _profiles_out = {}
        for _k, _v in self.profiles.items():
            _profiles_out[_k] = _v(*[self.objs[model_name][k] for k in self.parameters[_k]])

        # -- Fixing stellar density -- #
        if "stellar_density" in _profiles_out:
            stellar_density = _profiles_out["stellar_density"]
            del _profiles_out["stellar_density"]
        else:
            stellar_density = None

        # - attempting to generate the cluster - #
        if not hasattr(ClusterModel, self.load_method):
            raise ValueError(f"The load method {self.load_method} is not valid.")
        else:
            m = getattr(ClusterModel, self.load_method)(r_min, r_max, *list(_profiles_out.values()),
                                                        num_points=num_points, stellar_density=stellar_density,
                                                        gravity=gravity, **kwargs)

        return m

    def plot_summary(self,rmin=0.1,rmax=5000,npoints=5000,**kwargs):
        """
        Generates a plot gallery of all of the constituent profiles.
        Parameters
        ----------
        rmin: float
            The minimum radius to plot.
        rmax: float
            The maximum radius to plot.
        npoints: int
            The number of points to include.
        kwargs: dict, optional
            Additional key word arguments to pass through the function.

        Returns
        -------
        None
        """
        #  Setting up the arrays
        # ------------------------------------------------------------------------------------------------------------ #
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.geomspace(rmin,rmax,npoints)

        #  Setting up the figure.
        # ------------------------------------------------------------------------------------------------------------ #
        n_axes = len(self.profiles)
        _factors = []
        for i in range(1,n_axes+1):
            if n_axes % i == 0:
                _factors.append(i)

        nr,nc = _factors[len(_factors)//2],_factors[len(_factors)//2 - 1]
        fig,axes = plt.subplots(nr,nc,figsize=(5*nc,5.4*nr),sharex=True)

        #  Plotting
        # ------------------------------------------------------------------------------------------------------------ #

        for a,p in zip(axes.ravel(),self.profiles.items()):
            a.set_title(p[0])
            print(a,p)
            for k,v in self.objs.items():
                profile = p[1](*[v[j]  for j in self.parameters[p[0]]])
                y = profile(x)
                a.loglog(x, profile(x),**{_k:(h[k] if isinstance(h,dict) else h) for _k,h in kwargs.items()})

                if np.any(y<0):
                    a.set_yscale("symlog")



        plt.show()

class Vikhlinin06(ClusterCollection):
    """
    :py:class:`collection.ClusterCollection` instance representing the dataset from `Vikhlinin, A. et. al. 2006ApJ...640..691V <https://ui.adsabs.harvard.edu/abs/2006ApJ...640..691V/abstract>`_.
    """

    #  Dunder methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self):
        """
        Initializes the instance.
        """
        super().__init__(os.path.join(collections_directory, "Vikhlinin06.yaml"))

    @staticmethod
    def load():
        """
        Same as :py:meth:`collection.Vikhlinin06.__init__`, but as a static method.
        """
        return Vikhlinin06()

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def load_model(self, model_name, r_min=None, r_max=None, num_points=1000, gravity="Newtonian", **kwargs):
        """
        Loads the model with name ``model_name``.

        Parameters
        ----------
        model_name: str
            The name of the model to load.
        r_min: float, optional.
            The minimum radius at which to load. Defaults to ``rmin`` field in dataset.
        r_max: float, optional.
            The maximum radius at which to load. Defaults to ``r_det`` field.
        num_points: int
            The number of constituent points to use.
        gravity: str
            The gravity theory to use.
        **kwargs
            additional parameters to pass through the loading system.

        Returns
        -------
        ClusterModel
            The finished cluster model.
        """
        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if not r_min:
            r_min = self.objs[model_name]["r_min"]

        if not r_max:
            r_max = self.objs[model_name]["r_det"]

        return super().load_model(model_name, r_min, r_max, num_points=num_points, gravity=gravity, **kwargs)

class Ascasibar07(ClusterCollection):
    """
    :py:class:`collection.ClusterCollection` instance representing the dataset from `Ascasibar & Diego 2008MNRAS.383..369A <https://ui.adsabs.harvard.edu/abs/2008MNRAS.383..369A/abstract>`_.
    """

    #  Dunder methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self):
        """
        Initializes the instance.
        """
        super().__init__(os.path.join(collections_directory, "Ascasibar07.yaml"))

    @staticmethod
    def load():
        """
        Same as :py:meth:`collection.Ascasibar07.__init__`, but as a static method.
        """
        return Ascasibar07()

    #  Methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def load_model(self, model_name, r_min=None, r_max=None, num_points=1000, gravity="Newtonian", **kwargs):
        """
        Loads the model with name ``model_name``.

        Parameters
        ----------
        model_name: str
            The name of the model to load.
        r_min: float, optional.
            The minimum radius at which to load. Defaults to ``rmin`` field in dataset.
        r_max: float, optional.
            The maximum radius at which to load. Defaults to ``r_det`` field.
        num_points: int
            The number of constituent points to use.
        gravity: str
            The gravity theory to use.
        **kwargs
            additional parameters to pass through the loading system.

        Returns
        -------
        ClusterModel
            The finished cluster model.
        """
        #  Sanity Check
        # ------------------------------------------------------------------------------------------------------------ #
        if not r_min:
            r_min = self.objs[model_name]["r_min"]

        if not r_max:
            r_max = self.objs[model_name]["r_det"]

        return super().load_model(model_name, r_min, r_max, num_points=num_points, gravity=gravity, **kwargs)

class Sanderson10(ClusterCollection):
    """
    :py:class:`collection.ClusterCollection` instance representing the dataset from `A.J.R.Sanderson and T.J.Ponman 2010MNRAS.402...65S <https://ui.adsabs.harvard.edu/abs/2010MNRAS.402...65S/abstract>`_.
    """
    #  Dunder methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self):
        """
        Initializes the instance.
        """
        super().__init__(os.path.join(collections_directory, "Sanderson10.yaml"))

    @staticmethod
    def load():
        """
        Same as :py:meth:`collection.Vikhlinin10.__init__`, but as a static method.
        """
        return Sanderson10()



class CollectionsError(Exception):
    """
    Exception class for errors during dataset loading.
    """
    def __init__(self, message, section=None):
        self.message, self.status = message, section
        super().__init__(self.message)


# -------------------------------------------------------------------------------------------------------------------- #
#  Functions ========================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
def get_collections():
    """
    Returns the set of all of the available :py:class:`collection.ClusterCollection` objects saved to default.

    Returns
    -------
    list
        The available options.
    """
    out = []
    for file in os.listdir(collections_directory):
        if ".yaml" not in file:
            continue

        with open(os.path.join(collections_directory, file), "r") as f:
            ds = yaml.load(f, yaml.FullLoader)

        out.append((file,ds["global"]["name"], ds["global"]["description"]))

    return out


if __name__ == '__main__':
    import numpy as np
    u = Sanderson10.load()
    u.plot_summary(100,5000,c="red")

