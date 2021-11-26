import os
import unittest
import shutil
import requests
import zipfile
import logging
import subprocess
from tqdm import tqdm
from unittest import mock

from medcat.cli.config import get_auth_environment_vars, get_git_api_request_url, get_git_default_headers, set_git_global_git_credentials
from medcat.cli.system_utils import create_new_base_repository, force_delete_path, get_local_model_storage_path, make_orderer
from medcat.cli.package import package
from medcat.cli.download import download, get_matching_version

from medcat.cat import CAT

ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare

class PackageTests(unittest.TestCase):
    

    """
        Unit tests for git & utils, tests if folders can be created & accessed, as well as if the git credentials work etc.
    """

    tmp_full_model_tag_name_improved = "unit_test_model-1.1"
    unit_testing_model_path_origin = os.path.join(get_local_model_storage_path(), "_unit_test_model_tmp_")
    unit_testing_model_path_download_location = os.path.join(get_local_model_storage_path(), "_unit_test_model_")
    unit_test_specialist_model_name = "unit_test_specialist_model"
    unit_test_specialist_model_tag_name = "unit_test_specialist_model-1.0"
    unit_test_specialist_model_tag_name_improved = "unit_test_specialist_model-1.1"
    tmp_full_model_tag_name = "unit_test_model-1.0"
    
    set_git_global_git_credentials()

    @classmethod
    def setUpClass(cls):
        pass

    def get_input(text):
        return input(text)
   
    def get_test_model_name(test):
        return input(test)

    @ordered
    def test_download_files(self):
        logging.info("Attempting to download initial base models from the MedCAT repo with NO version history.")
        if not os.path.exists(self.unit_testing_model_path_download_location):
            os.mkdir(self.unit_testing_model_path_download_location)

        os.chdir(self.unit_testing_model_path_download_location)

        request_timeout = 1800 # seconds 

        url_cdb = "https://medcat.rosalind.kcl.ac.uk/media/cdb-medmen-v1.dat"
        url_vocab = "https://medcat.rosalind.kcl.ac.uk/media/vocab.dat"
        url_medcat_export_json = "https://raw.githubusercontent.com/CogStack/MedCAT/legacy/tutorial/data/MedCAT_Export.json"
        url_mc_status = "https://medcat.rosalind.kcl.ac.uk/media/mc_status.zip"
        
        cdb_file_name = url_cdb.split("/")[-1]
        vocab_file_name = url_vocab.split("/")[-1]
        medcat_export_file_name = url_medcat_export_json.split("/")[-1]
        mc_status_file_name = url_mc_status.split("/")[-1]

        cdb_f_path = os.path.join(self.unit_testing_model_path_download_location, cdb_file_name)
        vocab_f_path = os.path.join(self.unit_testing_model_path_download_location, vocab_file_name)
        medcat_export_f_path = os.path.join(self.unit_testing_model_path_download_location, medcat_export_file_name)
        mc_status_f_path = os.path.join(self.unit_testing_model_path_download_location, mc_status_file_name)

        r_get_cdb = requests.get(url_cdb, stream=True, allow_redirects=True, timeout=request_timeout)
        cdb_fsize = int(r_get_cdb.headers.get('content-length', 0))

        if (os.path.exists(cdb_f_path) and os.path.isfile(cdb_f_path) and os.path.getsize(cdb_f_path) >= cdb_fsize) is False:
            with tqdm(total=cdb_fsize, unit_scale=True, unit="iB", desc=cdb_file_name, initial=0, ascii=True) as pbar, \
                open(cdb_f_path, "wb") as cdb_file:
                for kbyte in r_get_cdb.iter_content(chunk_size=1024):
                    cdb_file.write(kbyte)
                    pbar.update(len(kbyte))
        else:
            logging.info("File already exists:" + cdb_f_path + ". No need to redownload.")

        # change cdb file name from "cdb*.dat" to "cdb.dat"
        shutil.copy(cdb_f_path, os.path.join(self.unit_testing_model_path_download_location, "cdb.dat"))    

        r_get_vocab = requests.get(url_vocab, stream=True, allow_redirects=True, timeout=request_timeout)
        vocab_fsize = int(r_get_vocab.headers.get('content-length', 0))

        if (os.path.exists(vocab_f_path) and os.path.isfile(vocab_f_path) and os.path.getsize(vocab_f_path) >= vocab_fsize) is False:
            with tqdm(total=vocab_fsize, unit_scale=True, unit="iB", desc=vocab_file_name, initial=0, ascii=True) as pbar, \
                open(vocab_f_path , "wb") as vocab_file:
                for kbyte in r_get_vocab.iter_content(chunk_size=1024):
                    vocab_file.write(kbyte)
                    pbar.update(len(kbyte))
        else:
            logging.info("File already exists:" + vocab_f_path + ". No need to redownload.")

        r_get_medcat_export = requests.get(url_medcat_export_json, stream=True, timeout=request_timeout)
        medcat_export_fsize = int(r_get_medcat_export.headers.get('content-length', 0))

        if (os.path.exists(medcat_export_f_path) and os.path.isfile(medcat_export_f_path) and \
            os.path.getsize(medcat_export_f_path) >= medcat_export_fsize) is False:
            with tqdm(total=medcat_export_fsize, unit_scale=True, unit="iB", desc=medcat_export_file_name, initial=0, ascii=True) as pbar, \
                open(medcat_export_f_path , "wb") as medcat_export_file:
                for kbyte in r_get_medcat_export.iter_content(chunk_size=1024):
                    medcat_export_file.write(kbyte)
                    pbar.update(len(kbyte))
        else:
            logging.info("File already exists:" + medcat_export_f_path + ". No need to redownload.")

        r_get_mc_status = requests.get(url_mc_status, stream=True, timeout=request_timeout)
        r_get_mc_status_fsize = int(r_get_mc_status.headers.get('content-length', 0))

        # MC STATUS FILE
        if (os.path.exists(mc_status_f_path) and os.path.isfile(mc_status_f_path) and \
            os.path.getsize(mc_status_f_path) >= r_get_mc_status_fsize) is False:
            with tqdm(total=r_get_mc_status_fsize, unit_scale=True, unit="iB", desc=mc_status_file_name, initial=0, ascii=True) as pbar, \
                open(mc_status_f_path , "wb") as mc_status_file:
                for kbyte in r_get_mc_status.iter_content(chunk_size=1024):
                    mc_status_file.write(kbyte)
                    pbar.update(len(kbyte))
        else:
            logging.info("File already exists:" + mc_status_f_path + ". No need to redownload.")

        with zipfile.ZipFile(mc_status_f_path, "r") as zip_file:
            zip_file.extractall(self.unit_testing_model_path_download_location)
            for name in zip_file.namelist():
                if name.endswith(("/", "\\")) and "meta" not in name:
                    if os.path.isdir(name) and not os.path.exists("meta_" + name):
                        os.rename(name, "meta_" + name)
   
        self.assertEqual(medcat_export_fsize, os.path.getsize(medcat_export_f_path))
        self.assertEqual(r_get_mc_status_fsize, os.path.getsize(mc_status_f_path))
        self.assertEqual(cdb_fsize, os.path.getsize(cdb_f_path))
        self.assertEqual(vocab_fsize, os.path.getsize(vocab_fsize))
    
    @ordered
    def test_standardize_model_files(self):
        print("Standardizing model files")
        if not os.path.exists(self.unit_testing_model_path_origin):
            os.mkdir(self.unit_testing_model_path_origin)
    
        os.chdir(self.unit_testing_model_path_download_location)
    
        cat = CAT.load(path="", vocab_input_file_name = "vocab.dat", cdb_input_file_name = "cdb.dat", trainer_data_file_name = "MedCAT_Export.json")
        
        os.chdir(self.unit_testing_model_path_origin)

        cat.save(vocab_output_file_name="vocab.dat", cdb_output_file_name="cdb.dat",\
                 trainer_data_file_name="MedCAT_Export.json", skip_stat_generation=False)
    
        self.assertEqual(os.path.exists(os.path.join(self.unit_testing_model_path_origin, "vocab.dat")), True)
        self.assertEqual(os.path.exists(os.path.join(self.unit_testing_model_path_origin, "cdb.dat")), True)
        self.assertEqual(os.path.exists(os.path.join(self.unit_testing_model_path_origin, "MedCAT_Export.json")), True)
        self.assertEqual(os.path.exists(os.path.join(self.unit_testing_model_path_origin, "meta_Status")), True)
    
   
    @ordered   
    def test_package_model_default(self):
        os.chdir(self.unit_testing_model_path_origin)
        with unittest.mock.patch('builtins.input', side_effect=["yes", "yes", "no", "yes"]):
            self.assertEqual(package(self.tmp_full_model_tag_name), True)
  
  
    @ordered     
    def test_package_model_default_improvement(self):
        new_release_tmp_path = os.path.join(get_local_model_storage_path(), "_unit_test_new_release_tmp_")

        if not os.path.exists(new_release_tmp_path):
            os.mkdir(new_release_tmp_path)
            
        os.chdir(new_release_tmp_path)

        cat = CAT.load(path="", full_model_tag_name=self.tmp_full_model_tag_name)
        text = ["My patient has kidney failure, bowel cancer and heart failure",
                "patient was evaluated by an ophthalmologist and diagnosed with conjunctivitis.\
                 patient was given eye drops that did not relieve her eye symptoms."]
                
        cat.train = True

        for sentence in text:
            doc_spacy = cat(sentence, do_train=True)

        cat.train = False

        print("AFTER TRAINING: ")

        cat.cdb.print_stats()
        cat.save()

        with unittest.mock.patch('builtins.input', side_effect=["yes", "yes", "yes", "yes"]):
            self.assertEqual(package(), True)
    
    @ordered   
    def test_download_base_unit_test_model(self):
        os.chdir(get_local_model_storage_path())
        with unittest.mock.patch('builtins.input', side_effect=["yes", "yes", "yes"]):
            self.assertEqual(download(self.tmp_full_model_tag_name), True)
    
    @ordered
    def test_x_package_model_specialist(self):

        new_release_tmp_path = os.path.join(get_local_model_storage_path(), "_unit_test_new_release_tmp_")

        if not os.path.exists(new_release_tmp_path):
            os.mkdir(new_release_tmp_path)
        os.chdir(new_release_tmp_path)

        cat = CAT.load(path="", full_model_tag_name=self.tmp_full_model_tag_name)

        text = ["My patient has pancreatitis, and liver disease.", "Patient has nausea, symptoms appeared, vomiting.",
                "Patient has coronavirus, symptoms appeared 2 weeks ago."]
        cat.train = True

        for sentence in text:
            doc_spacy = cat(sentence, do_train=True)
        cat.save()

        with unittest.mock.patch('builtins.input', side_effect=["No", "Yes", self.unit_test_specialist_model_name, "y"]):
            self.assertEqual(package(), True)
    
    @ordered
    def test_x_package_model_specialist_improved(self):
        
        new_release_tmp_path = os.path.join(get_local_model_storage_path(), "_unit_test_new_release_tmp_")
 
        force_delete_path(new_release_tmp_path)
 
        if not os.path.exists(new_release_tmp_path):
            os.mkdir(new_release_tmp_path)
 
        os.chdir(new_release_tmp_path)
 
        cat = CAT.load(path="", full_model_tag_name=self.unit_test_specialist_model_tag_name)
 
        text = ["My patient has Tuberculosis",
                "Tuberculosis, ulcer and gastroentritis + ToF."]
        cat.train = True
   
        for sentence in text:
            doc_spacy = cat(sentence, do_train=True)
   
        cat.train = False
   
        print("AFTER TRAINING:")
        cat.cdb.print_stats()
        cat.save()
 
        with unittest.mock.patch("builtins.input", side_effect=["y", "y", "y"]):
            self.assertEqual(package(), True)
  
    @classmethod
    def tearDownClass(cls):
       
        unit_test_dummy_repo_tear_down = os.path.join(get_local_model_storage_path(), "unit_test_dummy_repo_tear_down")
        
        if not os.path.exists(unit_test_dummy_repo_tear_down):
            os.mkdir(unit_test_dummy_repo_tear_down)

        create_new_base_repository(unit_test_dummy_repo_tear_down, git_repo_url=get_auth_environment_vars()["git_repo_url"])
        os.chdir(unit_test_dummy_repo_tear_down)

        tags_to_delete = [PackageTests.tmp_full_model_tag_name, PackageTests.tmp_full_model_tag_name_improved,
                        PackageTests.unit_test_specialist_model_tag_name, PackageTests.unit_test_specialist_model_tag_name_improved]

        for tag_name in tags_to_delete:
            release_id = get_matching_version(tag_name)["release_id"]
            delete_asset_url = get_git_api_request_url() + "releases/" + str(release_id)
            req_delete_release = requests.delete(url=delete_asset_url, headers=get_git_default_headers())

            if req_delete_release.status_code >= 400:
                logging.error("Response: " + str(req_delete_release.status_code)  + " Failed to delete release : " + str(release_id) + " " + tag_name )
                logging.error("Reason:" + req_delete_release)

            subprocess.run(["git", "push", "--delete", "origin", tag_name], cwd=unit_test_dummy_repo_tear_down)
        
        os.chdir(get_local_model_storage_path())

        for root, dirs, files in os.walk(get_local_model_storage_path()):
            if "unit_test" in root or "tutorial" in root:
                force_delete_path(root)
