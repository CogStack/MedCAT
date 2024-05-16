from typing import List, Dict, Any

import pkg_resources
import platform


def get_installed_packages() -> List[List[str]]:
    installed_packages = []
    for package in pkg_resources.working_set:
        installed_packages.append([package.project_name, package.version])
    return installed_packages


def get_environment_info() -> Dict[str, Any]:
    return {
        "dependencies": get_installed_packages(),
        "os": platform.platform(),
        "cpu_architecture": platform.machine(),
        "python_version": platform.python_version()
    }
