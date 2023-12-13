"""This module attempts to read the spacy compatibilty of
a model pack and (if necessary) compare it to the installed
spacy version.
"""
from typing import Tuple, List
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
