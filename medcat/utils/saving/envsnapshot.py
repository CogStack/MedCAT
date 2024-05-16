from typing import List, Dict, Any

import pkg_resources
import platform


ENV_SNAPSHOT_FILE_NAME = "environment_snapshot.json"


def get_installed_packages() -> List[List[str]]:
    """Get the installed packages and their versions.

    Returns:
        List[List[str]]: List of lists. Each item contains of a dependency name and version.
    """
    installed_packages = []
    for package in pkg_resources.working_set:
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
