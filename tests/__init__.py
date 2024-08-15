from typing import Callable, Tuple

from medcat.utils import decorators


class DeprecatedMethodCallException(ValueError):

    def __init__(self, func: Callable, msg: str,
                 depr_version: Tuple[int, int, int],
                 removal_version: Tuple[int, int, int]) -> None:
        super().__init__(f"A deprecated method {func.__name__} was called. Deprecation message:\n{msg}\n"
                         f"The method was deprecated in v{depr_version} and is scheduled for "
                         f"removal in v{removal_version}")


def deprecation_exception_raiser(message: str, depr_version: Tuple[int, int, int],
                     removal_version: Tuple[int, int, int]):
    def decorator(func: Callable) -> Callable:
        def wrapper(*_, **__):
            raise DeprecatedMethodCallException(func, message, depr_version, removal_version)
        return wrapper
    return decorator


decorators.deprecated = deprecation_exception_raiser
