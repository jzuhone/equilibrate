"""
Implemented collections of cluster models.
"""
import os
import pathlib as pt

import pandas as pd
import yaml
from halo import Halo

import cluster_generator.radial_profiles as rprofs
from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog, log_string

# -------------------------------------------------------------------------------------------------------------------- #
# Setup ============================================================================================================== #
# -------------------------------------------------------------------------------------------------------------------- #

# -- finding the base storage directory -- #
collections_directory = os.path.join(pt.Path(__file__).parents[0], "bin", ".collections")


class ClusterCollection:

    #  Dunder Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    def __init__(self, path):
        #  Loading from the path
        # ------------------------------------------------------------------------------------------------------------ #
        _fail_with_alert = False
        self.path = path
        with Halo(log_string(f"Loading collection from {path}...")) as halo:

            # -- loading the yaml data -- #
            try:
                with open(path, "r+") as yf:
                    _col_data = yaml.load(yf, yaml.FullLoader)
            except FileNotFoundError:
                halo.fail("No such file")
                raise FileNotFoundError(f"There is no {os.path.join(os.getcwd(), path)}. Is there a typo?")
            except yaml.YAMLError as yex:
                halo.fail("YAML error")
                raise yaml.YAMLError(
                    f"The yaml file at {os.path.join(os.getcwd(), path)} failed to load. message = {yex.__repr__()}")

            # -- pulling global data -- #
            if "global" not in _col_data:
                halo.fail(f"`global` key not in {path}.")
                raise CollectionsError(f"Failed to locate the 'global' key in {path}.", section="global")

            for attr in ["name", "description", "load_method"]:
                try:
                    setattr(self, attr, _col_data["global"][attr])
                except KeyError:
                    halo.fail(f"Missing key {attr} in globals.")
                    raise CollectionsError(f"Failed to locate `{attr}` key in globals for {path}.",
                                           section=f"global.{attr}")

            #  Managing profiles
            # -------------------------------------------------------------------------------------------------------- #

            # -- making sure they exist -- #
            if "profiles" not in _col_data["global"]:
                halo.fail("No profiles.")
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
                halo.warn(log_string(f"Loaded collection {self.name} with warnings."))
            else:
                halo.succeed(log_string(f"Loaded collection {self.name}"))

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
        return self._profiles

    @property
    def parameters(self):
        return self._parameters

    @property
    def names(self):
        return list(self.objs.keys())

    #  Methods
    # ----------------------------------------------------------------------------------------------------------------- #
    def load_model(self, model_name, r_min, r_max, num_points=1000, gravity="Newtonian", **kwargs):
        """
        Loads the model with name ``model_name``.

        Parameters
        ----------
        model_name: str
            The name of the model to load.
        r_min: float
            The minimum radius at which to load.
        r_max: float
            The maximum radius at which to load.
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


class Vikhlinin06(ClusterCollection):
    """
    The ClusterCollection associated with the Vikhlinin06 paper.
    """

    #  Dunder methods
    # ---------------------------------------------------------------------------------------------------------------- #
    def __init__(self):
        super().__init__(os.path.join(collections_directory, "Vikhlinin06.yaml"))

    @staticmethod
    def load():
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


class CollectionsError(Exception):
    def __init__(self, message, section=None):
        self.message, self.status = message, section
        super().__init__(self.message)


# -------------------------------------------------------------------------------------------------------------------- #
#  Functions ========================================================================================================= #
# -------------------------------------------------------------------------------------------------------------------- #
def get_collections():
    """
    Returns the set of all of the available ``ClusterCollection`` objects saved to default.

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

        out.append((ds["global"]["name"], ds["global"]["description"]))

    return out


if __name__ == '__main__':
    u = Vikhlinin06()
    u.load_model("A133", 10, 5000)
