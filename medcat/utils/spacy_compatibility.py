"""This module attempts to read the spacy compatibility of
a model pack and (if necessary) compare it to the installed
spacy version.
"""
from typing import Tuple, List, cast
import os
import re
from packaging import version
from packaging.specifiers import SpecifierSet

import spacy


SPACY_MODEL_REGEX = re.compile(r"(\w{2}_core_(\w{3,4})_(sm|md|lg|trf|xxl|\w+))|(spacy_model)")


def _is_spacy_model_folder(folder_name: str) -> bool:
    """Check if a folder within a model pack contains a spacy model.

    The idea is to do this without loading the model. That is because
    the version of the model may be incompatible with what we've got.
    And as such, loading may not be possible.

    Args:
        folder_name (str): The folder to check.

    Returns:
        bool: Whether the folder contains a spacy model.
    """
    # since we're trying to identify this solely from the
    # folder name, we only care about the base name.
    folder_name = os.path.basename(folder_name)
    if folder_name.startswith("meta_"):
        # these are MetaCat stuff (or should be)
        return False
    return bool(SPACY_MODEL_REGEX.match(folder_name))


def _find_spacy_model_folder(model_pack_folder: str) -> str:
    """Find the spacy model folder in a model pack folder.

    Args:
        model_pack_folder (str): The model pack folder

    Raises:
        ValueError: If it's ambiguous or there's no model folder.

    Returns:
        str: The full path to the model folder.
    """
    options: List[str] = []
    for folder_name in os.listdir(model_pack_folder):
        full_folder_path = os.path.join(model_pack_folder, folder_name)
        if not os.path.isdir(full_folder_path):
            continue
        if _is_spacy_model_folder(folder_name):
            options.append(full_folder_path)
    if len(options) != 1:
        raise ValueError("Unable to determine spacy folder name from "
                         f"{len(options)} ambiguous folders: {options}")
    return options[0]


def get_installed_spacy_version() -> str:
    """Get the spacy version installed currently.

    Returns:
        str: The currently installed spacy version.
    """
    return spacy.__version__


def get_installed_model_version(model_name: str) -> str:
    """Get the version of a model installed in spacy.

    Args:
        model_name (str): The model name.

    Returns:
        str: The version of the installed model.
    """
    if model_name not in spacy.util.get_installed_models():
        return 'N/A'
    # NOTE: I don't really know when spacy.info
    # might return a str instead
    return cast(dict, spacy.info(model_name))['version']


def _get_name_and_meta_of_spacy_model_in_medcat_modelpack(model_pack_path: str) -> Tuple[str, dict]:
    """Gets the name and meta information about a spacy model within a medcat model pack.

    PS: This gets the raw (folder) name of the spacy model.
        While this is usually (in models created after v1.2.4)
        identical to the spacy model version, that may not always
        be the case.

    Args:
        model_pack_path (str): The model pack path.

    Returns:
        Tuple[str, dict]: The name of the spacy model, and the meta information.
    """
    spacy_model_folder = _find_spacy_model_folder(model_pack_path)
    # NOTE: I don't really know when spacy.info
    # might return a str instead
    info = cast(dict, spacy.info(spacy_model_folder))
    return os.path.basename(spacy_model_folder), info


def get_name_and_version_of_spacy_model_in_medcat_modelpack(model_pack_path: str) -> Tuple[str, str, str]:
    """Get the name, version, and compatible spacy versions of a spacy model within a medcat model pack.

    PS: This gets the real name of the spacy model.
        While this is usually (in models created after v1.2.4)
        identical to the folder name, that may not always
        be the case.

    Args:
        model_pack_path (str): The model pack path.

    Returns:
        Tuple[str, str, str]: The name of the spacy model, its version, and supported spacy version.
    """
    _, info = _get_name_and_meta_of_spacy_model_in_medcat_modelpack(model_pack_path)
    true_name = info["lang"] + "_" + info['name']
    return true_name, info['version'], info["spacy_version"]


def _is_spacy_version_within_range(spacy_version_range: str) -> bool:
    """Checks whether the spacy version is within the specified range.

    The expected format of the version range is similar to that used
    in requirements and/or pip installs. E.g:
        - >=3.1.0,<3.2.0
        - ==3.1.0
        - >=3.1.0
        - <3.20

    Args:
        spacy_version_range (str): The requires spacy version range.

    Returns:
        bool: Whether the specified range is compatible.
    """
    spacy_version = version.parse(get_installed_spacy_version())
    range = SpecifierSet(spacy_version_range)
    return range.contains(spacy_version)


def medcat_model_pack_has_compatible_spacy_model(model_pack_path: str) -> bool:
    """Checks whether a medcat model pack has a spacy model compatible with installed spacy version.

    Args:
        model_pack_path (str): The model pack path.

    Returns:
        bool: Whether the spacy model in the model pack is compatible.
    """
    _, _, spacy_range = get_name_and_version_of_spacy_model_in_medcat_modelpack(model_pack_path)
    return _is_spacy_version_within_range(spacy_range)


def is_older_spacy_version(model_version: str) -> bool:
    """Checks if the specified version is older than the installed version.

    Args:
        model_version (str): The specified spacy version.

    Returns:
        bool: Whether the specified version is older.
    """
    installed_version = version.parse(get_installed_spacy_version())
    model_version = version.parse(model_version)
    return model_version <= installed_version


def medcat_model_pack_has_semi_compatible_spacy_model(model_pack_path: str) -> bool:
    """Checks whether the spacy model within a medcat model pack is
        compatible or older than the installed spacy version.

    This method returns `True` if the spacy model is compatible or
    released with a lower version number compared to the spacy
    version currently installed.

    We've found that most of the time older models will work with
    a newer version of spacy. Though there is a warning on spacy's
    side and they do not guarantee 100% compatibility, we've not
    seen issues so far.

    E.g for installed spacy 3.4.4 all the following will be suiable:
        - en_core_web_md-3.1.0
        - en_core_web_md-3.2.0
        - en_core_web_md-3.3.0
        - en_core_web_md-3.4.1
    However, for the same version, the following would not be suitable:
        - en_core_web_md-3.5.0
        - en_core_web_md-3.6.0
        - en_core_web_md-3.7.1

    Args:
        model_pack_path (str): The model pack path.

    Returns:
        bool: Whether the spacy model in the model pack is compatible.
    """
    (_,
     model_version,
     spacy_range) = get_name_and_version_of_spacy_model_in_medcat_modelpack(model_pack_path)
    if _is_spacy_version_within_range(spacy_range):
        return True
    return is_older_spacy_version(model_version)
