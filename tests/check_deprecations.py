from typing import List, Dict, Optional, Tuple, Callable
import ast
import os
from sys import argv as sys_argv
from sys import exit as sys_exit
from medcat.utils.decorators import deprecated


def get_decorator_args(decorator: ast.expr, decorator_name: str) -> Tuple[Optional[List[str]], Optional[Dict[str, str]]]:
    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == decorator_name:
        return decorator.args, {kw.arg: kw.value for kw in decorator.keywords}
    return None, None


def is_decorated_with(node: ast.FunctionDef, decorator_name: str) -> Tuple[bool, List[str], Dict[str, str]]:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == decorator_name:
            return True, [], {}
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == decorator_name:
            args, kwargs = get_decorator_args(decorator, decorator_name)
            return True, args, kwargs
    return False, [], {}


class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, decorator_name: str):
        self.decorator_name = decorator_name
        self.decorated_functions: List[Dict[str, Optional[List[str]]]] = []
        self.context: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.context.append(node.name)
        is_decorated, args, kwargs = is_decorated_with(node, self.decorator_name)
        if is_decorated:
            self.decorated_functions.append({
                'name': '.'.join(self.context),
                'args': args,
                'kwargs': kwargs
            })
        self.generic_visit(node)
        self.context.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()


def find_decorated_functions_in_file(filepath: str, decorator_name: str) -> List[Dict[str, Optional[List[str]]]]:
    with open(filepath, "r") as source:
        tree = ast.parse(source.read())

    visitor = FunctionVisitor(decorator_name)
    visitor.visit(tree)
    return visitor.decorated_functions


def find_decorated_functions_in_codebase(codebase_path: str, decorator_name: str) -> Dict[str, List[Dict[str, Optional[List[str]]]]]:
    decorated_functions: Dict[str, List[Dict[str, Optional[List[str]]]]] = {}
    for root, _, files in os.walk(codebase_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                decorated_funcs = find_decorated_functions_in_file(filepath, decorator_name)
                if decorated_funcs:
                    decorated_functions[filepath] = decorated_funcs
    return decorated_functions


def extract_version_from_tuple(tuple_node: ast.Tuple) -> Tuple[int, int, int]:
    """Extract constant values from an ast.Tuple node.

    Args:
        tuple_node (ast.Tuple): The AST node representing the tuple.

    Raises:
        ValueError: If the tuple contains unsuitable values.

    Returns:
        Tuple[int, int, int]: The major, minor, and patch version.
    """
    values = []
    for element in tuple_node.elts:
        if isinstance(element, ast.Constant):
            cur_value = element.value
        else:
            raise ValueError(f"Unsupported element type in tuple: {type(element)}")
        values.append(cur_value)
        if not isinstance(cur_value, int):
            raise ValueError(f"Unknown type of value in version tuple: {type(cur_value)}: {cur_value}")
    if len(values) != 3:
        raise ValueError(f"Unexpected number of version elements ({len(values)}): {values}")
    return tuple(values)


def get_deprecated_methods_that_should_have_been_removed(codebase_path: str,
                                                         decorator_name: str,
                                                         medcat_version: Tuple[int, int, int]
                                                         ) -> List[Tuple[str, str, Tuple[int, int, int]]]:
    """Get deprecated methods that should have been removed.

    Args:
        codebase_path (str): Path to codebase.
        decorator_name (str): Name of decorator to check.
        medcat_version (Tuple[int, int, int]): The current MedCAT version.

    Returns:
        List[Tuple[str, str, Tuple[int, int, int]]]:
            The list of file, method, and version in which the method should have been deprecated.
    """
    decorated_functions = find_decorated_functions_in_codebase(codebase_path, decorator_name)

    should_be_removed = []
    for filepath, funcs in decorated_functions.items():
        for func in funcs:
            func_name = func['name']
            args, kwargs = func['args'], func['kwargs']
            if 'removal_version' in kwargs:
                rem_ver = kwargs['removal_version']
            else:
                rem_ver = args[-1]
            rem_ver = extract_version_from_tuple(rem_ver)
            if rem_ver <= medcat_version:
                should_be_removed.append((filepath, func_name, rem_ver))
    return should_be_removed


def _ver2str(ver: Tuple[int, int, int]) -> str:
    maj, min, patch = ver
    return f"v{maj}.{min}.{patch}"


def main(args: List[str] = sys_argv[1:],
         deprecated_decorator: Callable[[], Callable] = deprecated):
    decorator_name = deprecated_decorator.__name__
    pos_args = [arg for arg in args if not arg.startswith("-")]
    codebase_path = 'medcat' if len(pos_args) <= 1 else pos_args[1]
    print("arg0", repr(args[0]))
    remove_ver_prefix = '--remove-prefix' in args
    pure_ver = pos_args[0]
    if remove_ver_prefix:
        # remove v from (e.g) v1.12.0
        pure_ver = pure_ver[1:]
    medcat_version = tuple(int(s) for s in pure_ver.split("."))
    compare_next_minor_release = '--next-version' in args

    # pad out medcat varesions
    # NOTE: Mostly so that e.g (1, 12, 0) <= (1, 12, 0) would be True.
    #       Otherwise (1, 12, 0) <= (1, 12) would equate to False.
    if len(medcat_version) < 3:
        medcat_version = tuple(list(medcat_version) + [0,] * (3 - len(medcat_version)))
    # NOTE: In main GHA workflow we know the current minor release
    #       but after that release has been done, we (generally, but not always!)
    #       want to start removing deprecated methods due to be removed before
    #       the next minor release.
    if compare_next_minor_release:
        l_ver = list(medcat_version)
        l_ver[1] += 1
        medcat_version = tuple(l_ver)

    to_remove = get_deprecated_methods_that_should_have_been_removed(codebase_path, decorator_name, medcat_version)

    ver_descr = "next" if compare_next_minor_release else "current"
    for filepath, func_name, rem_ver in to_remove:
        print("SHOULD ALREADY BE REMOVED")
        print(f"In file: {filepath}")
        print(f" Method: {func_name}")
        print(f" Scheduled for removal in: {_ver2str(rem_ver)} ({ver_descr} version: {_ver2str(medcat_version)})")
    if to_remove:
        print("Found issues - see above")
        sys_exit(1)


if __name__ == "__main__":
    main()
