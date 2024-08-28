from typing import List, Dict, Any, Set

import os
import re
import pkg_resources
import platform
import logging


logger = logging.getLogger(__name__)


ENV_SNAPSHOT_FILE_NAME = "environment_snapshot.json"

INSTALL_REQUIRES_FILE_PATH = os.path.join(os.path.dirname(__file__),
                                          "..", "..", "..",
                                          "install_requires.txt")
# NOTE: The install_requires.txt file is copied into the wheel during build
#       so that it can be included in the distributed package.
#       However, that means it's 1 folder closer to this file since it'll now
#       be in the root of the package rather than the root of the project.
INSTALL_REQUIRES_FILE_PATH_PIP = os.path.join(os.path.dirname(__file__),
                                              "..", "..",
                                              "install_requires.txt")


def get_direct_dependencies() -> Set[str]:
    """Get the set of direct dependency names.

    The current implementation reads install_requires.txt for dependenceies,
    removes comments, whitespace, quotes; removes the versions and returns
    the names as a set.

    Returns:
        Set[str]: The set of direct dependency names.
    """
    req_file = INSTALL_REQUIRES_FILE_PATH
    if not os.path.exists(req_file):
        # When pip-installed. See note above near constant definition
        req_file = INSTALL_REQUIRES_FILE_PATH_PIP
    with open(req_file) as f:
        # read every line, strip quotes and comments
        dep_lines = [line.split("#")[0].replace("'", "").replace('"', "").strip() for line in f.readlines()]
        # remove comment-only (or empty) lines
        deps = [dep for dep in dep_lines if dep]
    return set(re.split("[@<=>~]", dep)[0].strip() for dep in deps)


def _update_installed_dependencies_recursive(
        gathered: Dict[str, str],
        package: pkg_resources.Distribution) -> Dict[str, str]:
    if package.project_name in gathered:
        logger.debug("Trying to update already found transitive dependency '%'",
                     package.egg_name)
        return gathered
    for req in package.requires():
        if req.project_name in gathered:
            logger.debug("Trying to look up already found transitive dependency '%'",
                         req.project_name)
            continue # don't look for it again
        try:
            dep = pkg_resources.get_distribution(req.project_name)
        except pkg_resources.DistributionNotFound as e:
            logger.warning("Unable to locate requirement '%s':", req.project_name,
                           exc_info=e)
            continue
        gathered[dep.project_name] = dep.version
        _update_installed_dependencies_recursive(gathered, dep)
    return gathered


def get_transitive_deps(direct_deps: List[str]) -> List[List[str]]:
    """Get the transitive dependencies of the direct dependencies.

    Args:
        direct_deps (List[str]): List of direct dependencies.

    Returns:
        List[List[str]]: The list of dependency names along with their versions.
    """
    # map from name to version so as to avoid multiples of the same package
    all_transitive_deps: Dict[str, str] = {}
    for dep in direct_deps:
        package = pkg_resources.get_distribution(dep)
        _update_installed_dependencies_recursive(all_transitive_deps, package)
    trans_deps = [list(td) for td in all_transitive_deps.items()]
    return sorted(trans_deps, key=lambda t: t[0])


def get_installed_packages() -> List[List[str]]:
    """Get the installed packages and their versions.

    Returns:
        List[List[str]]: List of lists. Each item contains of a dependency name and version.
    """
    direct_deps = get_direct_dependencies()
    installed_packages = []
    for package in pkg_resources.working_set:
        if package.project_name not in direct_deps:
            continue
        installed_packages.append([package.project_name, package.version])
    return installed_packages


def get_environment_info(include_transitive_deps: bool = True) -> Dict[str, Any]:
    """Get the current environment information.

    Args:
        include_transitive_deps (bool): Whether to include transitive dependencies. Defaults to True.

    This includes dependency versions, the OS, the CPU architecture and the python version.

    Returns:
        Dict[str, Any]: _description_
    """
    env_info = {
        "dependencies": get_installed_packages(),
        "os": platform.platform(),
        "cpu_architecture": platform.machine(),
        "python_version": platform.python_version()
    }
    if include_transitive_deps:
        direct_deps = [dep_name for dep_name, _ in env_info["dependencies"]]
        env_info["transitive_deps"] = get_transitive_deps(direct_deps)
    return env_info
