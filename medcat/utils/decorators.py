import warnings
import functools
from typing import Callable, Tuple


def _format_version(ver: Tuple[int, int, int]) -> str:
    return ".".join(str(v) for v in ver)


def deprecated(message: str, depr_version: Tuple[int, int, int], removal_version: Tuple[int, int, int]) -> Callable:
    """Deprecate a method.

    Args:
        message (str): The deprecation message.
        depr_version (Tuple[int, int, int]): The first version of MedCAT where this was deprecated.
        removal_version (Tuple[int, int, int]): The first version of MedCAT where this will be removed.

    Returns:
        Callable: _description_
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> Callable:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("Function {} has been deprecated.{}".format(func.__name__, " " + message if message else ""))
            warnings.warn(f"The above function was deprecated in v{_format_version(depr_version)} "
                          f"and will be removed in v{removal_version}")
            warnings.simplefilter("default", DeprecationWarning)
            return func(*args, **kwargs)
        return wrapped
    return decorator


def check_positive(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapped(*args, **kwargs) -> Callable:
        for idx, arg in enumerate(args):
            if isinstance(arg, int):
                if arg < 1:
                    raise ValueError("Argument at position %s is not a positive integer" % idx)
        for key, value in kwargs.items():
            if isinstance(value, int) and value < 1:
                raise ValueError("Argument '%s' is not a positive integer" % key)
        return func(*args, **kwargs)
    return wrapped
