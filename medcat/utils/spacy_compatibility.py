"""This module attempts to read the spacy compatibilty of
a model pack and (if necessary) compare it to the installed
spacy version.
"""
from typing import Tuple, List, cast
import os
import re

import spacy


SPACY_MODEL_REGEX = re.compile(r"(\w{2}_core_(\w{3,4})_(sm|md|lg|trf|xxl|\w+))|(spacy_model)")


def is_spacy_model_folder(folder_name: str) -> bool:
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


def find_spacy_model_folder(model_pack_folder: str) -> str:
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
        if is_spacy_model_folder(folder_name):
            options.append(full_folder_path)
    if len(options) != 1:
        raise ValueError("Unable to determine spacy folder name from "
                         f"{len(options)} ambiguous folders: {options}")
    return options[0]


def get_installed_spacy_version() -> str:
    """Get the spacy version installed currently.

    Returns:
        str: The currently installed spacy verison.
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


def get_name_and_meta_of_spacy_model_in_medcat_modelpack(model_pack_path: str) -> Tuple[str, dict]:
    """Gets the name and meta information about a spacy model within a medcat model pack.

    Args:
        model_pack_path (str): The model pack path.

    Returns:
        Tuple[str, dict]: The name of the spacy model, and the meta information.
    """
    spacy_model_folder = find_spacy_model_folder(model_pack_path)
    # NOTE: I don't really know when spacy.info
    # might return a str instead
    info = cast(dict, spacy.info(spacy_model_folder))
    return os.path.basename(spacy_model_folder), info


def get_name_and_version_of_spacy_model_in_medcat_modelpack(model_pack_path: str) -> Tuple[str, str]:
    name, info = get_name_and_meta_of_spacy_model_in_medcat_modelpack(model_pack_path)
    return name, info['version']
