from typing import List, Dict, Any, Set

import re
import pkg_resources
import platform
from importlib_metadata import distribution


ENV_SNAPSHOT_FILE_NAME = "environment_snapshot.json"


def get_direct_dependencies() -> Set[str]:
    """Get the set of direct dependeny names.

    The current implementation uses importlib_metadata to figure out
    the names of the required packages and removes their version info.

    Raises:
        ValueError: If the unlikely event that the dependencies are unable to be obtained.

    Returns:
        Set[str]: The set of direct dependeny names.
    """
    package_name = __package__.split(".")[0]
    dist = distribution(package_name)
    deps = dist.metadata.get_all('Requires-Dist')
    if not deps:
        raise ValueError("Unable to identify dependencies")
    return set(re.split("[<=>~]", dep)[0] for dep in deps)


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
