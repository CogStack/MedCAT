from typing import Tuple

from medcat.cat import CAT

SemanticVersion = Tuple[int, int, int]


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
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Unknown version string: {version_str}")
    return tuple([int(part) for part in parts])


def get_semantic_version(cat: CAT) -> SemanticVersion:
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
