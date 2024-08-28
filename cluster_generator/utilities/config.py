"""Cluster generator configuration management / utilities."""
import operator
import os
from functools import reduce
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable

from ruamel.yaml import (
    YAML,
    CommentedMap,
    Loader,
    MappingNode,
    Node,
    ScalarNode,
    SequenceNode,
)
from unyt import unyt_array, unyt_quantity

config_directory = os.path.join(Path(__file__).parents[1], "bin", "config.yaml")
""" str: The system directory where the ``cluster_generator`` configuration is stored.

The underlying ``.yaml`` file may be altered by the user to set configuration values.
"""


# YAML CONSTRUCTORS
# This section of code defines specialized YAML constructors so that we can enable
# yaml loading and dumping of non-python standard library types.
def unyt_array_constructor(loader: "Loader", node: "Node") -> unyt_array:
    """Custom constructor function for :py:class:`unyt.unyt_array`.

    Parameters
    ----------
    loader: :py:class:`Loader`
        The YAML loader.
    node: :py:class:`Node`
        The node to construct.

    Returns
    -------
    unyt.unyt_array
        The unyt array constructed.

    Notes
    -----
    We allow two formats, either the entry is an array with mixed types so that the final
    entry is the unit or it is a dictionary with an ``array`` and ``unit`` key / value pairing.
    """
    if isinstance(node, ScalarNode):
        raise ValueError(
            f"Cannot reconstruct an unyt_array object from a ScalarNode {ScalarNode}."
        )
    elif isinstance(node, SequenceNode):
        # Assume format is [x_0,x_1,..., unit]
        sequence = loader.construct_sequence(node)

        try:
            return unyt_array(sequence[:-1], sequence[-1])
        except Exception:
            raise ValueError(
                f"Failed to construct unyt_array from YAML sequence {sequence}. (NODE={node})."
            )
    elif isinstance(node, MappingNode):
        mapping = loader.construct_mapping(node)

        try:
            array = mapping["array"]
            unit = mapping["unit"]
        except KeyError:
            raise KeyError(
                f"YAML node {node} should have keys array and unit to be parsed as unyt_array."
            )

        return unyt_array(array, unit)

    else:
        raise TypeError(f"Type {type(node)} not supported.")


def unyt_quantity_constructor(loader: "Loader", node: "Node") -> unyt_quantity:
    """Custom constructor function for :py:class:`unyt.unyt_quantity`.

    Parameters
    ----------
    loader: :py:class:`Loader`
        The YAML loader.
    node: :py:class:`Node`
        The node to construct.

    Returns
    -------
    unyt.unyt_array
        The unyt array constructed.
    """
    if isinstance(node, ScalarNode):
        scalar = loader.construct_scalar(node)
        if not isinstance(scalar, str):
            raise ValueError(
                f"Cannot parse unyt_quantity from scalar of type {type(scalar)}."
            )

        return unyt_quantity.from_string(scalar)

    elif isinstance(node, SequenceNode):
        # Assume format is [x_0,x_1,..., unit]
        sequence = loader.construct_sequence(node)

        if not len(sequence) == 2:
            raise ValueError(
                f"{node} has length {len(sequence)}, which cannot be parsed as unyt_quantity."
            )

        return unyt_quantity(sequence[0], sequence[1])

    elif isinstance(node, MappingNode):
        mapping = loader.construct_mapping(node)
        try:
            value = mapping["value"]
            unit = mapping["unit"]
        except KeyError:
            raise KeyError(
                f"YAML node {node} should have keys value and unit to be parsed as unyt_value."
            )

        return unyt_quantity(value, unit)

    else:
        raise TypeError(f"Type {type(node)} not supported.")


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""

    pass


class YAMLConfig:
    """
    Configuration class for managing YAML-based configuration files.

    This class provides methods to load, modify, save, and validate configuration
    settings stored in a YAML file. It supports lazy loading, automatic saving, nested
    structures, environment variable overrides, and comment preservation. Additionally,
    it provides built-in support for custom YAML constructors to handle specialized data types.

    Attributes
    ----------
    path : Path
        The YAML file represented by this class instance.
    yaml : YAML
        The YAML loader instance configured with custom constructors.
    """

    _yaml_loaders: ClassVar[dict[str, Callable[["Loader", "Node"], Any]]] = {
        "!unyt_arr": unyt_array_constructor,
        "!unyt_qty": unyt_quantity_constructor,
    }

    def __init__(self, path: str | Path, base_loader: "YAML" = None):
        """
        Initialize a YAML configuration around a specific path.

        Parameters
        ----------
        path : str or Path
            The path to the underlying YAML file.
        base_loader : YAML, optional
            A custom YAML loader instance. If not provided, a default loader is created.

        Raises
        ------
        ConfigurationError
            If the configuration file does not exist at the specified path.
        """
        self.path: Path = Path(path)
        """ Path: The YAML file represented by this class instance."""

        # Setup the YAML with the correct loaders.
        if base_loader is None:
            self.yaml = YAML(typ="safe")
        else:
            self.yaml = base_loader

        for loader_tag, loader in self.__class__._yaml_loaders.items():
            self.yaml.constructor.add_constructor(loader_tag, loader)

        # Setup private storage variable for the underlying dictionary
        # to permit lazy loading of the data.
        self._config: CommentedMap | None = None

    def __repr__(self):
        return f"YAMLConfig(path={self.path!r})"

    def __str__(self):
        return f"YAMLConfig(name={self.path.name})"

    def __getitem__(self, key: str | Iterable[str]):
        return self.get(key)

    def __setitem__(self, key: str | Iterable[str], value: Any):
        self.set(key, value)

    def __delitem__(self, key: str | Iterable[str]):
        self.delete(key)

    def __contains__(self, key: str | Iterable[str]):
        keys = key.split(".") if isinstance(key, str) else key
        try:
            reduce(operator.getitem, keys, self.config)
            return True
        except KeyError:
            return False

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if key in {"path", "_config", "yaml"}:
            super().__setattr__(key, value)
        else:
            self[key] = value

    def _load(self) -> CommentedMap:
        """Load the configuration from the YAML file."""
        if not self.path.exists():
            raise ConfigurationError(f"Configuration file not found at {self.path}")

        try:
            with self.path.open("r") as file:
                # If we don't get anything from the file (it's empty or just comments), we just
                # return a blank commented map.
                return self.yaml.load(file) or CommentedMap()
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")

    def _save(self) -> None:
        """Save the configuration to the YAML file."""
        try:
            with self.path.open("w") as file:
                self.yaml.dump(self._config, file)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")

    @property
    def config(self) -> CommentedMap:
        """Access the configuration, loading it from the file if necessary."""
        if self._config is None:
            self._config = self._load()
        return self._config

    def get(self, keys: str | Iterable[str], default: Any = None) -> Any:
        """
        Get a configuration value using a list of keys or a dot-separated string.

        Parameters
        ----------
        keys : str or Iterable[str]
            The key path to retrieve the value.
        default : Any, optional
            The default value to return if the key does not exist. By default, raises a KeyError.

        Returns
        -------
        Any
            The configuration value corresponding to the provided key(s).

        Raises
        ------
        KeyError
            If the key does not exist and no default is provided.
        """
        if isinstance(keys, str):
            keys = keys.split(".")
        else:
            keys = list(keys)

        try:
            return reduce(operator.getitem, keys, self.config)
        except KeyError:
            return default

    def set(self, keys: str | Iterable[str], value: Any, save: bool = True) -> None:
        """
        Set a configuration value using a list of keys or a dot-separated string.

        Parameters
        ----------
        keys : str or Iterable[str]
            The key path to set the value.
        value : Any
            The value to set.
        save : bool, optional
            Whether to save the configuration after setting the value. Default is True.

        Returns
        -------
        None
        """
        keys = keys.split(".") if isinstance(keys, str) else keys
        sub_config = self.config

        for key in keys[:-1]:
            sub_config = sub_config.setdefault(key, {})

        sub_config[keys[-1]] = value

        if save:
            self._save()

    def delete(self, keys: str | Iterable[str], save: bool = True) -> None:
        """
        Delete a configuration value using a list of keys or a dot-separated string.

        Parameters
        ----------
        keys : str or Iterable[str]
            The key path to delete the value.
        save : bool, optional
            Whether to save the configuration after deleting the value. Default is True.

        Returns
        -------
        None
        """
        keys = keys.split(".") if isinstance(keys, str) else keys
        sub_config = self.config

        for key in keys[:-1]:
            sub_config = sub_config[key]

        del sub_config[keys[-1]]

        if save:
            self._save()

    def reload(self) -> None:
        """
        Reload the configuration from the YAML file.

        Clears the cached configuration and reloads it from disk.

        Returns
        -------
        None
        """
        self._config = None
        _ = self.config  # Re-load configuration


cgparams: YAMLConfig = YAMLConfig(config_directory)
""" YAMLConfig: The ``cluster_generator`` configuration object."""
