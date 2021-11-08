import os
import unittest
from medcat.cli.system_utils import *
from medcat.cli.config import *

class UntilTests(unittest.TestCase):
    
    """
        Unit tests for git & utils, tests if folders can be created & accessed, as well as if the git credentials work etc.
    """
    set_git_global_git_credentials()

    def setUp(self):
        self.tmp_full_model_tag_name = "unit_test_model-1.0"
        self.new_model_package_folder = os.path.join(get_local_model_storage_path(), self.tmp_full_model_tag_name)
    
    def test_access_config(self):
        self.assertNotEqual(get_auth_environment_vars(), False)

    def test_models_folder_access(self):
        result = get_local_model_storage_path()
        self.assertNotEqual(result, "")
   
    def test_create_model_tag_folder(self):
        if(os.path.exists(self.tmp_full_model_tag_name)):
            force_delete_path(self.new_model_package_folder)
        self.assertEqual(create_model_folder(self.tmp_full_model_tag_name), True)

    def test_create_new_base_repository(self):
        self.assertEqual(create_new_base_repository(repo_folder_path=self.new_model_package_folder, git_repo_url=get_auth_environment_vars()["git_repo_url"]), True)

    def test_model_tag_folder_is_git_repository(self):
        self.assertEqual(is_dir_git_repository(self.new_model_package_folder), True)