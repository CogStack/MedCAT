from typing import Tuple
import re
import os

import dill

from medcat.cat import CAT

SemanticVersion = Tuple[int, int, int]


# Regex as per:
# https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
SEMANTIC_VERSION_REGEX = (r"^(0|[1-9]\d*)"  # major
                          r"\.(0|[1-9]\d*)"  # .minor
                          r"\.(0|[1-9]\d*)"  # .patch
                          # and then some trailing stuff
                          f"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
                          f"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$")
SEMANTIC_VERSION_PATTERN = re.compile(SEMANTIC_VERSION_REGEX)


def get_semantic_version(version: str) -> SemanticVersion:
    """Get the semantiv version from the string.

    Args:
        version (str): The version string.

    Raises:
        ValueError: If the version string does not match the semantic versioning format.

    Returns:
        SemanticVersion | Tuple[int, int, int]: The major, minor and patch version
    """
    match = SEMANTIC_VERSION_PATTERN.match(version)
    if not match:
        raise ValueError(f"Unknown version string: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def get_version_from_modelcard(d: dict) -> SemanticVersion:
    """Gets the the major.minor.patch version from a model card.

    The version needs to be specified at:
        model_card["MedCAT Version"]
    The version is expected to be semantic (major.minor.patch).

    Args:
        d (dict): The model card in dict format.

    Returns:
        SemanticVersion | Tuple[int, int, int]: The major, minor and patch version
    """
    version_str: str = d["MedCAT Version"]
    return get_semantic_version(version_str)


def get_semantic_version_from_model(cat: CAT) -> SemanticVersion:
    """Get the semantic version of a CAT model.

    This uses the `get_version_from_modelcard` method on the model's
    model card.

    So it is equivalen to `get_version_from_modelcard(cat.get_model_card(as_dict=True))`.

    Args:
        cat (CAT): The CAT model.

    Returns:
        SemanticVersion | Tuple[int, int, int]: The major, minor and patch version
    """
    return get_version_from_modelcard(cat.get_model_card(as_dict=True))


def get_version_from_cdb_dump(cdb_path: str) -> SemanticVersion:
    """Get the version from a CDB dump (cdb.dat).

    The version information is expected in the following location:
        cdb["config"]["version"]["medcat_version"]

    Args:
        cdb_path (str): The path to cdb.dat

    Returns:
        SemanticVersion | Tuple[int, int, int]: The major, minor and patch version
    """
    with open(cdb_path, 'rb') as f:
        d = dill.load(f)
    config: dict = d["config"]
    version = config["version"]["medcat_version"]
    return get_semantic_version(version)


def get_version_from_modelpack_zip(zip_path: str, cdb_file_name="cdb.dat") -> SemanticVersion:
    """Get the semantic version from a MedCAT model pack zip file.

    This involves simply reading the config on file and reading the version information from there.

    The zip file is extracted if it has not yet been extracted.

    Args:
        zip_path (str): The zip file path for the model pack.
        cdb_file_name (str, optional): The CDB file name to use. Defaults to "cdb.dat".

    Returns:
        SemanticVersion | Tuple[int, int, int]: The major, minor and patch version
    """
    model_pack_path = CAT.attempt_unpack(zip_path)
    return get_version_from_cdb_dump(os.path.join(model_pack_path, cdb_file_name))
