from typing import List, Dict, Any, Set

import os
import re
import pkg_resources
import platform


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
    """Get the set of direct dependeny names.

    The current implementation reads install_requires.txt for dependenceies,
    removes comments, whitespace, quotes; removes the versions and returns
    the names as a set.

    Returns:
        Set[str]: The set of direct dependeny names.
    """
    req_file = INSTALL_REQUIRES_FILE_PATH
    if not os.path.exists(req_file):
        # When pip-installed. See note above near constant definiation
        req_file = INSTALL_REQUIRES_FILE_PATH_PIP
    with open(req_file) as f:
        # read every line, strip quotes and comments
        dep_lines = [line.split("#")[0].replace("'", "").replace('"', "").strip() for line in f.readlines()]
        # remove comment-only (or empty) lines
        deps = [dep for dep in dep_lines if dep]
    return set(re.split("[@<=>~]", dep)[0].strip() for dep in deps)


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


def get_environment_info() -> Dict[str, Any]:
    """Get the current environment information.

    This includes dependency versions, the OS, the CPU architecture and the python version.

    Returns:
        Dict[str, Any]: _description_
    """
    return {
        "dependencies": get_installed_packages(),
        "os": platform.platform(),
        "cpu_architecture": platform.machine(),
        "python_version": platform.python_version()
    }
