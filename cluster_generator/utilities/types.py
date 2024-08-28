"""Special types and type hinting utilities."""
from abc import ABC, abstractmethod
from numbers import Number
from typing import Collection, Generic, Mapping, Type, TypeVar

from more_itertools import always_iterable
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from unyt import Unit, unyt_array, unyt_quantity

try:
    from typing import Self  # noqa
except ImportError:
    from typing_extensions import Self as Self  # noqa

# Defining commonly used type annotations so that they do not need
# to be redefined elsewhere whenever they are used.
Instance = TypeVar("Instance")
Value = TypeVar("Value")
Attribute = TypeVar("Attribute")

# Type annotations for types which must have explicit units.
HasUnits_Scalar = unyt_quantity
HasUnits_Vector = unyt_array
HasUnits = HasUnits_Scalar | HasUnits_Vector

# Type annotation for types which may carry implicit or explicit units.
MaybeUnits_Scalar = HasUnits_Scalar | Number
MaybeUnits_Vector = HasUnits_Vector | NDArray[Number]
MaybeUnits = MaybeUnits_Scalar | MaybeUnits_Vector

T = TypeVar("T")


class Registry(ABC, Generic[T]):
    """
    Abstract base class for creating registries for specific types of objects.

    Subclasses should define the `CLASS_REGISTRY_OBJECT` attribute to specify
    the type of objects they manage.

    """

    CLASS_REGISTRY_OBJECT: Type[T]
    """ The class this registry wraps around."""

    def __init__(self, initial_data: Mapping[str, T] = None, **kwargs: T):
        """
        Initialize the registry and optionally populate it with initial data.

        Parameters
        ----------
        initial_data: Mapping[str, T], optional
            A dictionary or mapping containing initial data to register. Keys are names
            and values are the objects to register.

        kwargs: T
            Additional objects to register. Keys are names and values are the objects to register.

        Raises
        ------
        TypeError
            If any object is not an instance of `CLASS_REGISTRY_OBJECT`.
        """
        self._registry: dict[str, T] = {}

        # Register initial data if provided
        if initial_data:
            for name, obj in initial_data.items():
                self.add(name, obj)

        # Register additional objects passed as kwargs
        for name, obj in kwargs.items():
            self.add(name, obj)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(registered_objects={list(self._registry.keys())})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(objects={len(self._registry)})"

    def __getattr__(self, name: str) -> T:
        """
        Allows registered objects to be accessed as attributes.

        Parameters
        ----------
        name: str
            The name of the registered object.

        Returns
        -------
        T
            The registered object.

        Raises
        ------
        AttributeError
            If the object with the given name is not registered.
        """
        try:
            return self._registry[name]
        except KeyError:
            raise AttributeError(f"No object named '{name}' is registered.")

    def add(self, name: str, obj: T) -> None:
        """
        Add a new object to the registry.

        Parameters
        ----------
        name: str
            The name to register the object under.
        obj: T
            The object to register.

        Raises
        ------
        TypeError
            If the object is not an instance of `CLASS_REGISTRY_OBJECT`.
        """
        if not isinstance(obj, self.CLASS_REGISTRY_OBJECT):
            raise TypeError(
                f"Object must be an instance of {self.CLASS_REGISTRY_OBJECT.__name__}, got {type(obj).__name__} instead."
            )
        if name in self._registry.keys():
            raise ValueError(f"Object named '{name}' is already registered.")

        self._registry[name] = obj
        self.validate_object(name, obj)

    def remove(self, name: str) -> None:
        """
        Remove an object from the registry.

        Parameters
        ----------
        name: str
            The name of the object to remove.

        Raises
        ------
        KeyError
            If no object with the given name is registered.
        """
        if name in self._registry:
            del self._registry[name]
        else:
            raise KeyError(f"No object named '{name}' is registered.")

    def get(self, name: str) -> T:
        """
        Get a registered object by its name.

        Parameters
        ----------
        name: str
            The name of the registered object.

        Returns
        -------
        T
            The registered object.

        Raises
        ------
        KeyError
            If no object with the given name is registered.
        """
        return self._registry[name]

    @abstractmethod
    def validate_object(self, name: str, obj: T) -> None:
        """
        Optional method for additional validation when an object is added to the registry.

        Subclasses can override this method to implement custom validation logic.

        Parameters
        ----------
        name: str
            The name of the registered object.
        obj: T
            The object to validate.

        Raises
        ------
        ValueError
            If the object is invalid according to the subclass's criteria.
        """
        pass


def ensure_ytquantity(
    x: Number | unyt_quantity, default_units: Unit | str
) -> unyt_quantity:
    """Ensure that an input ``x`` is a unit-ed quantity with the expected units.

    Parameters
    ----------
    x: Any
        The value to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_quantity
        The corresponding quantity with correct units.
    """
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


def ensure_ytarray(x: ArrayLike, default_units: Unit | str) -> unyt_array:
    """Ensure that an input ``x`` is a unit-ed array with the expected units.

    Parameters
    ----------
    x: Any
        The values to enforce units on.
    default_units: Unit
        The unit expected / to be applied if missing.

    Returns
    -------
    unyt_array
        The corresponding array with correct units.
    """
    if isinstance(x, unyt_array):
        return x.to(default_units)
    elif isinstance(x, tuple) and len(x) == 2:
        return unyt_array(*x).to(default_units)
    else:
        return unyt_array(x, default_units)


def ensure_list(x: Collection) -> list:
    """Convert generic iterable to list."""
    return list(always_iterable(x))
