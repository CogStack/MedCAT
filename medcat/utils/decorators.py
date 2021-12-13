import warnings
import functools
from typing import Callable


def deprecated(message: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(*args, **kwargs) -> Callable:
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("Function {} has been deprecated.{}".format(func.__name__, " " + message if message else ""))
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
