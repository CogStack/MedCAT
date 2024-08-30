from typing import Tuple, List
import re
import os
import shutil
import argparse
import logging
from functools import partial

import dill
import json

from medcat.cat import CAT
from medcat.utils.decorators import deprecated
from medcat.utils.config_utils import default_weighted_average

logger = logging.getLogger(__name__)

SemanticVersion = Tuple[int, int, int]


# Regex as per:
# https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string
SEMANTIC_VERSION_REGEX = (r"^(0|[1-9]\d*)"  # major
                          r"\.(0|[1-9]\d*)"  # .minor
                          # CHANGE FROM NORM - allowing dev before patch version number
                          # but NOT capturing the group
                          r"\.(?:dev)?"
                          r"(0|[1-9]\d*)"  # .patch
                          # and then some trailing stuff
                          r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
                          r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$")
SEMANTIC_VERSION_PATTERN = re.compile(SEMANTIC_VERSION_REGEX)


CDB_FILE_NAME = "cdb.dat"


def get_semantic_version(version: str) -> SemanticVersion:
    """Get the semantiv version from the string.

    Args:
        version (str): The version string.

    Raises:
        ValueError: If the version string does not match the semantic versioning format.

    Returns:
        SemanticVersion: The major, minor and patch version
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
        SemanticVersion: The major, minor and patch version
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
        SemanticVersion: The major, minor and patch version
    """
    return get_version_from_modelcard(cat.get_model_card(as_dict=True))


def get_version_from_cdb_dump(cdb_path: str) -> SemanticVersion:
    """Get the version from a CDB dump (cdb.dat).

    The version information is expected in the following location:
        cdb["config"]["version"]["medcat_version"]

    Args:
        cdb_path (str): The path to cdb.dat

    Returns:
        SemanticVersion: The major, minor and patch version
    """
    with open(cdb_path, 'rb') as f:
        d = dill.load(f)
    config: dict = d["config"]
    version = config["version"]["medcat_version"]
    return get_semantic_version(version)


def get_version_from_modelpack_zip(zip_path: str, cdb_file_name=CDB_FILE_NAME) -> SemanticVersion:
    """Get the semantic version from a MedCAT model pack zip file.

    This involves simply reading the config on file and reading the version information from there.

    The zip file is extracted if it has not yet been extracted.

    Args:
        zip_path (str): The zip file path for the model pack.
        cdb_file_name (str, optional): The CDB file name to use. Defaults to "cdb.dat".

    Returns:
        SemanticVersion: The major, minor and patch version
    """
    model_pack_path = CAT.attempt_unpack(zip_path)
    return get_version_from_cdb_dump(os.path.join(model_pack_path, cdb_file_name))


UPDATE_VERSION = (1, 3, 0)


class ConfigUpgrader:
    """Config updater.

    Attempts to upgrade pre 1.3.0 medcat configs to the newer format.

    Args:
        zip_path (str): The model pack zip path.
        cdb_file_name (str, optional): The CDB file name. Defaults to "cdb.dat".
    """

    def __init__(self, zip_path: str, cdb_file_name: str = CDB_FILE_NAME) -> None:
        self.model_pack_path = CAT.attempt_unpack(zip_path)
        self.cdb_path = os.path.join(self.model_pack_path, cdb_file_name)
        self.current_version = get_version_from_cdb_dump(self.cdb_path)
        logger.debug("Loaded model from %s at version %s",
                     self.model_pack_path, self.current_version)

    def needs_upgrade(self) -> bool:
        """Check if the specified modelpack needs an upgrade.

        It needs an upgrade if its version is less than 1.3.0.

        Returns:
            bool: Whether or not an upgrade is needed.
        """
        return self.current_version < UPDATE_VERSION

    def _get_relevant_files(self, ignore_hidden: bool = True) -> List[str]:
        """Get the list of relevant files with full path names.

        By default this will ignore hidden files (those that start with '.').

        Args:
            ignore_hidden (bool): Whether to ignore hidden files. Defaults to True.

        Returns:
            List[str]: The list of relevant file names to copy.
        """
        return [os.path.join(self.model_pack_path, fn)  # ignores hidden files
                for fn in os.listdir(self.model_pack_path) if (ignore_hidden and not fn.startswith("."))]

    def _check_existence(self, files_to_copy: List[str], new_path: str, overwrite: bool):
        if overwrite:
            return  # ignore all
        if not os.path.exists(new_path):
            os.makedirs(new_path)
            return  # all good, new folder
        # check file existence in new (existing) path
        for file_to_copy in files_to_copy:
            new_file_name = os.path.join(
                new_path, os.path.basename(file_to_copy))
            if os.path.exists(new_file_name):
                raise ValueError(f"File already exists: {new_file_name}. "
                                 "Pass overwrite=True to overwrite")

    def _copy_files(self, files_to_copy: List[str], new_path: str) -> None:
        for file_to_copy in files_to_copy:
            new_file_name = os.path.join(
                new_path, os.path.basename(file_to_copy))
            if os.path.isdir(file_to_copy):
                # if exists is OK since it should have been checked before
                # if it was not to be overwritten
                logger.debug("Copying folder %s to %s",
                             file_to_copy, new_file_name)
                shutil.copytree(file_to_copy, new_file_name,
                                dirs_exist_ok=True)
            else:
                logger.debug("Copying file %s to %s",
                             file_to_copy, new_file_name)
                shutil.copy(file_to_copy, new_file_name)

    def upgrade(self, new_path: str, overwrite: bool = False) -> None:
        """Upgrade the model.

        The upgrade copies all the files from the original folder
        to the new folder.

        After copying, it changes the config into the format
        required by MedCAT after version 1.3.0.

        Args:
            new_path (str): The path for the new model pack folder.
            overwrite (bool): Whether to overwrite new path. Defaults to False.

        Raises:
            ValueError: If one of the target files exists and cannot be overwritten.
            IncorrectModel: If model pack does not need an upgrade
        """
        if not self.needs_upgrade():
            raise IncorrectModel(f"Model pack does not need upgrade: {self.model_pack_path} "
                                 f"since it's at version: {self.current_version}")
        logger.info("Starting to upgrade %s at (version %s)",
                    self.model_pack_path, self.current_version)
        files_to_copy = self._get_relevant_files()
        try:
            self._check_existence(files_to_copy, new_path, overwrite)
        except ValueError as e:
            raise e
        logger.debug("Copying files from %s", self.model_pack_path)
        self._copy_files(files_to_copy, new_path)
        logger.info("Going to try and fix CDB")
        self._fix_cdb(new_path)
        self._make_archive(new_path)

    def _fix_cdb(self, new_path: str) -> None:
        new_cdb_path = os.path.join(new_path, os.path.basename(self.cdb_path))
        with open(new_cdb_path, 'rb') as f:
            data = dill.load(f)
        # make the changes

        logger.debug("Fixing CDB issue #1 (linking.filters.cui)")
        # Number 1
        # the linking.filters.cuis is set to "{}"
        # which is assumed to be an empty set, but actually
        # evaluates to an empty dict instead
        cuis = data['config']['linking']['filters']['cuis']
        if cuis == {}:
            # though it _should_ be the empty set
            data['config']['linking']['filters']['cuis'] = set(cuis)
        # save modified version
        logger.debug("Saving CDB back into %s", new_cdb_path)
        with open(new_cdb_path, 'wb') as f:
            dill.dump(data, f)

    def _make_archive(self, new_path: str):
        logger.debug("Taking data from %s and writing it to %s.zip",
                     new_path, new_path)
        shutil.make_archive(
            base_name=new_path, format='zip', base_dir=new_path)


def parse_args() -> argparse.Namespace:
    """Parse the arguments from the CLI.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", help="The action. Currently, only 'fix-config' or 'allow-pre-1.12' are available.",
        choices=['fix-config', 'allow-pre-1.12'], type=str.lower)
    parser.add_argument("modelpack", help="MedCAT modelpack zip path")
    parser.add_argument("newpath", help="The path for the new modelpack")
    parser.add_argument(
        "--overwrite", help="Allow overvwriting existing files", action="store_true")
    parser.add_argument(
        "--silent", help="Disable logging", action="store_true")
    parser.add_argument(
        "--verbose", help="Show debug output", action="store_true")
    return parser.parse_args()


def setup_logging(args: argparse.Namespace) -> None:
    """Setup logging for the runnable based on CLI arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    if not args.silent:
        logger.addHandler(logging.StreamHandler())
    if args.verbose:
        logger.setLevel(logging.DEBUG)


@deprecated("This is no longer needed. Since medcat 1.10 (PR #352) "
            "this dealt with automatically upon model load.",
            depr_version=(1, 10, 0), removal_version=(1, 14, 0))
def fix_config(args: argparse.Namespace) -> None:
    """Perform the fix-config action based on the CLI arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    logger.debug("Setting up upgrader")
    upgrader = ConfigUpgrader(args.modelpack)
    logger.debug("Starting the upgrade process")
    upgrader.upgrade(args.newpath, overwrite=args.overwrite)


def _do_pre_1_12_fix(model_pack_path: str) -> CAT:
    cat = CAT.load_model_pack(model_pack_path)
    waf = cat.cdb.weighted_average_function
    is_def = waf is default_weighted_average
    is_partial = (isinstance(waf, partial)
                 and waf.func is default_weighted_average)
    if is_def:
        factor = 0.0004
        logger.info("Was using default weighted average")
    elif is_partial:
        pargs = waf.args
        pkwargs = waf.keywords
        factor = pargs[0] if pargs else pkwargs['factor']
        logger.info("Was using a (near) default weighted average")
    else:
        raise IncorrectModel("Model does not have fixable weighted_average tied to its CDB, "
                             f"found: {waf}")
    cat.cdb.weighted_average_function = lambda step: max(0.1, 1 - (step ** 2 * factor))
    return cat


def _set_change(val: dict):
    return {"py/set": val["==SET=="]}


def _pattern_change(val: dict):
    return {
        "py/object": "re.Pattern",
        "pattern": val["==PATTERN=="]
    }


TO_CHANGE = {
    "preprocessing.words_to_skip": _set_change,
    "preprocessing.keep_punct": _set_change,
    "preprocessing.do_not_normalize": _set_change,
    "linking.filters.cuis": _set_change,
    "linking.filters.cuis_exclude": _set_change,
    "word_skipper": _pattern_change,
    "punct_checker": _pattern_change,
}


def _fix_config_for_pre_1_12(folder: str):
    config_path = os.path.join(folder, 'config.json')
    with open(config_path) as f:
        data = json.load(f)
    for fix_path, fixer in TO_CHANGE.items():
        logger.info("[Pre 1.12 fix] Changing %s", fix_path)
        cur_path = fix_path
        last_dict = data
        while "." in cur_path:
            cur_key, cur_path = cur_path.split(".", 1)
            last_dict = last_dict[cur_key]
        last_key = cur_path
        last_value = last_dict[last_key]
        last_dict[last_key] = fixer(last_value)
    logger.info("[Pre 1.12 fix] Saving config back to %s", config_path)
    with open(config_path, 'w') as f:
        json.dump(data, f)
    logger.info("[Pre 1.12 fix] Recreating archive for %s", folder)
    shutil.make_archive(folder, 'zip', root_dir=folder)


@deprecated("This is only really needed for 1.12+ models "
            "to be converted to lower versions of medcat. "
            "It should not be needed in the long run.",
            depr_version=(1, 13, 0), removal_version=(1, 14, 0))
def allow_loading_with_pre_1_12(args: argparse.Namespace):
    """This method converts a model created after medcat 1.12
    such that it can be loaded in previous versions.

    The main two things it does:
    - Simplifies the weighted average function attached to the CDB.
    - Makes the config json-compatible

    Expected / used arguments in CLI:
    - modelpack: The input model pack path
    - newpath: The output model pack path
    - overwrite: Whether to overwrite the new model

    Raises:
        ValueError: If the file already exists

    Args:
        args (argparse.Namespace): The CLI arguments.
    """
    # this will fix the weighted_average function if possible
    # since 1.12 this is within the CDB and generally refers
    # to a method on medcat.utils.config_utils and the method
    # and/or the module do not exist in previous version
    cat = _do_pre_1_12_fix(args.modelpack)
    if not args.overwrite and os.path.exists(args.newpath):
        raise ValueError(f"File already exists: {args.newpath}. "
                          "Set --overwrite to overwrite")
    mpn = cat.create_model_pack(args.newpath)
    full_path = os.path.join(args.newpath, mpn)
    logger.info("Saving model to: %s", full_path)
    # now that the model has saved, we also need to do make 
    # some changes to the config to allow it to be properly
    # loaded by jsonpickle (used before 1.12) rather than
    # just json (used by 1.12+)
    _fix_config_for_pre_1_12(full_path)


class IncorrectModel(ValueError):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def main() -> None:
    """Run the CLI associated with this module.

    Raises:
        ValueError: If an unknown action is provided.
    """
    args = parse_args()
    setup_logging(args)
    logger.debug("Will attempt to perform action %s", args.action)
    if args.action == 'fix-config':
        fix_config(args)
    elif args.action == 'allow-pre-1.12':
        allow_loading_with_pre_1_12(args)
    else:
        raise ValueError(f"Unknown action: {args.action}")


if __name__ == "__main__":
    main()
