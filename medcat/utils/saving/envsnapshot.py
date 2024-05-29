from typing import List, Dict, Any, Set

import os
import re
import pkg_resources
import platform


ENV_SNAPSHOT_FILE_NAME = "environment_snapshot.json"
SETUP_PY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "setup.py"))
SETUP_PY_REGEX = re.compile("install_requires=\[([\s\S]*?)\]")


def get_direct_dependencies() -> Set[str]:
    """Get the set of direct dependeny names.

    The current implementation reads setup.py for the install_requires
    keyword argument, evaluates the list, removes the versions and returns
    the names as a set.

    Raises:
        FileNotFoundError: If the setup.py file was not found.
        ValueError: If found different sets of instal lrequirements.

    Returns:
        Set[str]: The set of direct dependeny names.
    """
    if not os.path.exists(SETUP_PY_PATH):
        raise FileNotFoundError(f"{SETUP_PY_PATH} does not exist.")
    with open(SETUP_PY_PATH) as f:
        setup_py_code = f.read()
    found = SETUP_PY_REGEX.findall(setup_py_code)
    if not found:
        raise ValueError("Did not find install requirements in setup.py")
    if len(found) > 1:
        raise ValueError("Ambiguous install requirements in setup.py")
    deps_str = found[0]
    # evaluate list of dependencies (including potential version pins)
    deps: List[str] = eval("[" + deps_str + "]")
    # remove versions where applicable
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
