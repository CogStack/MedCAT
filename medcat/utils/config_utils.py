from functools import partial
from typing import Callable, Optional, Protocol, Dict, Any
import logging
from pydantic import BaseModel


class WAFCarrier(Protocol):

    @property
    def weighted_average_function(self) -> Callable[[float], int]:
        pass


logger = logging.getLogger(__name__)


def is_old_type_config_dict(d: dict) -> bool:
    if set(('py/object', 'py/state')) <= set(d.keys()):
        return True
    return False


def fix_waf_lambda(carrier: WAFCarrier) -> None:
    weighted_average_function = carrier.weighted_average_function  # type: ignore
    if callable(weighted_average_function) and getattr(weighted_average_function, "__name__", None) == "<lambda>":
        # the following type ignoring is for mypy because it is unable to detect the signature
        carrier.weighted_average_function = partial(weighted_average, factor=0.0004) # type: ignore


# NOTE: This method is a hacky workaround. The type ignores are because I cannot
#       import config here since it would produce a circular import
def ensure_backward_compatibility(config: BaseModel, workers: Callable[[], int]) -> None:
    # Hacky way of supporting old CDBs
    if hasattr(config.linking, 'weighted_average_function'):  # type: ignore
        fix_waf_lambda(config.linking)  # type: ignore
    if config.general.workers is None:  # type: ignore
        config.general.workers = workers()  # type: ignore
    disabled_comps = config.pre_load.spacy_disabled_components  # type: ignore
    if 'tagger' in disabled_comps and 'lemmatizer' not in disabled_comps:
        config.pre_load.spacy_disabled_components.append('lemmatizer')  # type: ignore


def get_and_del_weighted_average_from_config(config: BaseModel) -> Optional[Callable[[int], float]]:
    if not hasattr(config, 'linking'):
        return None
    linking = config.linking
    if not hasattr(linking, 'weighted_average_function'):
        return None
    waf = linking.weighted_average_function
    delattr(linking, 'weighted_average_function')
    return waf


def weighted_average(step: int, factor: float) -> float:
    return max(0.1, 1 - (step ** 2 * factor))


def default_weighted_average(step: int) -> float:
    return weighted_average(step, factor=0.0004)


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


def remap_nested_dict(current_dict: Dict[str, Any], mappings: Dict[str, Dict[str, str]],
                      in_place: bool = True) -> Dict[str, Any]:
    """Remaps nested dict according to the specified mappings.

    Can do so in existing dict (i.e change it) or in a brand new dict.

    If using a brand new dict, only keys specified in the mappings will exist in new dict.
    Otherwise it will retain other keys as well.

    The mappings describe for each target (new) parent path:
    - Each new path maps to the old path (e.g `a.a1.a12`)

    Args:
        current_dict (Dict[str, Any]): Input nested dict.
        mappings (Dict[str, Dict[str, str]]): The mappings.
        in_place (bool): Whether to do this in existing dict. Defaults to True.

    Returns:
        Dict[str, Any]: Existing dict or new one.
    """
    target_dict = current_dict if in_place else {}
    for target_key, key_mappings in mappings.items():
        if target_key not in target_dict:
            target_dict[target_key] = {}
        cur_top_target = target_dict[target_key]
        for sub_path, source_path in key_mappings.items():
            source_value = current_dict
            source_keys = source_path.split('.')
            for nr, key in enumerate(source_keys):
                if nr == len(source_keys) - 1 and in_place:
                    source_value = source_value.pop(key)
                else:
                    source_value = source_value[key]
            cur_top_target[sub_path] = source_value
    return target_dict


# for each section name (in the new config)
#  - maps the new name to the path to the old name
#  - only specifies changed paths
CONFIG_REMAP_MAPPINGS = {
    'pre_load': {
        'spacy_model': 'general.spacy_model',
        'spacy_disabled_components': 'general.spacy_disabled_components',
        'log_level': 'general.log_level',
        'log_format': 'general.log_format',
        'log_path': 'general.log_path',
        # preprocessing
        'preprocessing_stopwords': 'preprocessing.stopwords',
        'max_document_length': 'preprocessing.max_document_length',
    }
}
"""Remapping from v1.11 (inclusive) to newer config structure for main config"""


def legacy_remap_mct_config(config_dict: dict) -> dict:
    """Maps the nested dict config from old format to new.

    The method changes the values within the input dict as needed.

    This refers to the format change after version 1.11.

    Args:
        config_dict (dict): The original nested dict.

    Returns:
        dict: The same (changed) dict
    """
    return remap_nested_dict(config_dict, CONFIG_REMAP_MAPPINGS)
