from functools import partial
from typing import Callable
import logging


logger = logging.getLogger(__name__)


def weighted_average(step: int, factor: float) -> float:
    return max(0.1, 1 - (step ** 2 * factor))


def attempt_fix_weighted_average_function(waf: Callable[[int], float]
                                          ) -> Callable[[int], float]:
    """Attempf fix weighted_average_function.

    When saving a model (dill.dump) in older python versions (3.10 and before)
    and then loading it back up with newer versions of python (3.11 and later)
    there can be an issue with loading `config.linking.weighted_average_function`.
    The value attributed to this is generally a `functools.partial`. And it loads
    just fine, but cannot subsequently be called.

    This method fixes the issue if the default function is used. It retains
    the arguments and keyword arguments of the original `partial.

    What this method does is recreate the partial function based on the
    arguments originally provided. Along with the fix, it logs a warning.

    However, if a non-default method is used, we are unable to fix it.
    That is because we do not know which method may have been used.
    In that case, a warning is logged.

    Args:
        waf (Callable[[int], float]): The weighted average function.

    Returns:
        Callable[[int], float]: The (potentially) fixed function.
    """
    try:
        waf(1)
        return waf
    except TypeError:
        # this means we need to apply the fix
        return _fix_waf(waf)


def _fix_waf(waf):
    if not str(waf.func).startswith("<function weighted_average at "):
        logger.warning("It seems the value of "
                        "`config.linking.weighted_average_function` "
                        "in the config does not work properly. While we "
                        "are aware of the issue and know how to fix the "
                        "default value, doing so for arbitrary methods "
                        "is not trivial. This is the case we've found. "
                        "The method does not seem to work properly, but "
                        "it has a non-default value so we are unable to "
                        "perform a fix for it. This is more than likely "
                        "to cause the an error when running the pipe. "
                        "To fix this, change the value of "
                        "`config.linking.weighted_average_function` "
                        "manually before using the CAT instance")
        return waf
    logging.warning("Fixing config.linking.weighted_average_function "
                    "since the value saved does not work properly. "
                    "This is usually due to having loaded a model "
                    "that was originally saved in older versions of "
                    "python and thus something has gone wrong when "
                    "loading the method. This fix should not affect "
                    "usage, but if you wish to avoid the warning "
                    "you may want to save the model pack again using "
                    "a newer version of python (3.11 or later).")
    return partial(weighted_average, *waf.args, **waf.keywords)
