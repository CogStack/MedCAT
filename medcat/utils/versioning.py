from typing import Tuple
import re

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
