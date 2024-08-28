"""Utilities for documentation management."""
import functools
import warnings
from typing import Callable


def _setup_deprecation_message(version, alternative, reason):
    if version is not None:
        version_specifier = f" as of {version}"
    else:
        version_specifier = ""

    if alternative is not None:
        alternative_specifier = f" {alternative} should be used instead."
    else:
        alternative_specifier = ""

    if reason is not None:
        reason_specifier = f" {reason}"
    else:
        reason_specifier = ""

    return f"%(function)s is deprecated{version_specifier}{reason_specifier}.{alternative_specifier}"


def deprecated(
    version_deprecated: str = None, alternative: Callable = None, reason: str = None
) -> Callable[[Callable], Callable]:
    """Mark a function or class as deprecated.

    Parameters
    ----------
    version_deprecated: str, optional
        The version at which the particular callable was deprecated.
    alternative: callable, optional
        The alternative function to use.
    reason: str, optional
        The reason for the deprecation.
    """
    # Setting up the message.
    _message_template = _setup_deprecation_message(
        version_deprecated,
        alternative.__name__ if alternative is not None else None,
        reason,
    )

    def decorator(func: Callable) -> Callable:
        # Add the deprecation to the documentation.
        if func.__doc__ is not None:
            func.__doc__ += f"\n.. deprecated:: {version_deprecated}\n   {reason}."
        else:
            func.__doc__ = f"\n.. deprecated:: {version_deprecated}\n   {reason}."

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn(
                _message_template % dict(function=func.__name__),
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return new_func

    return decorator
