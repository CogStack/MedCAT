from typing import Any
import platform
import os
import tempfile
import json
import zipfile

from medcat.cat import CAT
from medcat.utils.saving import envsnapshot

import unittest


def list_zip_contents(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        return zip_ref.namelist()


class DirectDependenciesTests(unittest.TestCase):

    def setUp(self) -> None:
        self.direct_deps = envsnapshot.get_direct_dependencies()

    def test_nonempty(self):
        self.assertTrue(self.direct_deps)

    def test_does_not_contain_versions(self, version_starters: str = '<=>~'):
        for dep in self.direct_deps:
            for vs in version_starters:
                with self.subTest(f"DEP '{dep}' check for '{vs}'"):
                    self.assertNotIn(vs, dep)

    def test_deps_are_installed_packages(self):
        for dep in self.direct_deps:
            with self.subTest(f"Has '{dep}'"):
                envsnapshot.pkg_resources.require(dep)


class EnvSnapshotAloneTests(unittest.TestCase):

    def setUp(self) -> None:
        self.env_info = envsnapshot.get_environment_info()

    def test_info_is_dict(self):
        self.assertIsInstance(self.env_info, dict)

    def test_info_is_not_empty(self):
        self.assertTrue(self.env_info)

    def assert_has_target(self, target: str, expected: Any):
        self.assertIn(target, self.env_info)
        py_ver = self.env_info[target]
        self.assertEqual(py_ver, expected)

    def test_has_os(self):
        self.assert_has_target("os", platform.platform())

    def test_has_py_ver(self):
        self.assert_has_target("python_version", platform.python_version())

    def test_has_cpu_arch(self):
        self.assert_has_target("cpu_architecture", platform.machine())

    def test_has_dependencies(self, name: str = "dependencies"):
        # NOTE: just making sure it's a anon-empty list
        self.assertIn(name, self.env_info)
        deps = self.env_info[name]
        self.assertTrue(deps)

    def test_all_direct_dependencies_are_installed(self):
        deps = self.env_info['dependencies']
        direct_deps = envsnapshot.get_direct_dependencies()
        self.assertEqual(len(deps), len(direct_deps))


CAT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples")
ENV_SNAPSHOT_FILE_NAME = envsnapshot.ENV_SNAPSHOT_FILE_NAME


class EnvSnapshotInCATTests(unittest.TestCase):
    expected_env = envsnapshot.get_environment_info()

    @classmethod
    def setUpClass(cls) -> None:
        cls.cat = CAT.load_model_pack(CAT_PATH)
        cls._temp_dir = tempfile.TemporaryDirectory()
        mpn = cls.cat.create_model_pack(cls._temp_dir.name)
        cls.cat_folder = os.path.join(cls._temp_dir.name, mpn)
        cls.envrion_file_path = os.path.join(cls.cat_folder, ENV_SNAPSHOT_FILE_NAME)

    def test_has_environment(self):
        self.assertTrue(os.path.exists(self.envrion_file_path))

    def test_eviron_saved(self):
        with open(self.envrion_file_path) as f:
            saved_info: dict = json.load(f)
        self.assertEqual(saved_info.keys(), self.expected_env.keys())
        for k in saved_info:
            with self.subTest(k):
                v1, v2 = saved_info[k], self.expected_env[k]
                self.assertEqual(v1, v2)

    def test_zip_has_env_snapshot(self):
        filenames = list_zip_contents(self.cat_folder + ".zip")
        self.assertIn(ENV_SNAPSHOT_FILE_NAME, filenames)


class EnvSnapshotTranistiveDepsTests(unittest.TestCase):
    TRANSITIVE_KEY = "transitive_deps"
    env_no_transitive = envsnapshot.get_environment_info(
        include_transitive_deps=False)
    env_w_transitive = envsnapshot.get_environment_info(
        include_transitive_deps=True)

    def test_can_exclude_transitive(self):
        self.assertNotIn(self.TRANSITIVE_KEY, self.env_no_transitive)

    def test_can_have_transitive(self):
        self.assertIn(self.TRANSITIVE_KEY, self.env_w_transitive)

    @property
    def transitive_deps(self):
        return self.env_w_transitive[self.TRANSITIVE_KEY]

    def test_can_find_transitive(self):
        self.assertTrue(self.transitive_deps) # more than one
        self.assertIsInstance(self.transitive_deps, list)

    def test_trans_deps_have_name_and_versions(self):
        for i, td in enumerate(self.transitive_deps):
            with self.subTest(f"Transitive dependency - {i}"):
                self.assertIsInstance(td, list)
                self.assertEqual(len(td), 2)
                self.assertIsInstance(td[0], str)
                self.assertIsInstance(td[1], str)
