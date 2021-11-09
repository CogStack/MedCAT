import shutil
import stat
import subprocess
import git
import os
import sys
import dill
import pickle
import logging
import traceback
import re

import json
import numpy

from dataclasses import asdict

from medcat.cli.modeltagdata import ModelTagData
from medcat.cli.modelstats import TrainerStats

if os.name == "nt":
    import win32api, win32con


def load_file_from_model_storage(full_model_tag_name="", file_name="", model_folder=".", bypass_model_path=False, ignore_non_model_files=False):
    """
        Looks into the models directory in your ~/.cache/medcat/model_name/ folder.

        :param full_model_tag_name: self-explanatory
        :param file_name: the model file name that we want to load e.g: "vocab.dat", "cdb.dat", "MedCAT_export.json" etc
        :param model_folder: path to model folder, the default is medcat's package folder,
                             use bypass_model_path=True if you wish to specify your own path
        :param bypass_model_path: will look into specified folder instead of model folder

        :param ignore_non_model_files: this is used in the detection and injection of model tag data,
                                       essentially some files are not actualy classes therefore they cannot hold any version data 
                                       although they are part of the model as a whole for example modelcard.md or .npy files

        :return: file data
    """

    local_model_folder_path = False 

    if full_model_tag_name != "":
        local_model_folder_path = get_downloaded_local_model_folder(full_model_tag_name)

    full_file_path = False

    if local_model_folder_path:
        full_file_path = os.path.join(local_model_folder_path, file_name)

    if bypass_model_path is True:
        full_file_path = os.path.join(model_folder, file_name)

    data = False
        
    if type(full_file_path) is str and "MedCAT_Export.json" in full_file_path:
        with open(full_file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        
            if "trainer_stats" not in data.keys():
                data["trainer_stats"] = asdict(TrainerStats())
            
            if "vc_model_tag_data" not in data.keys():
                data["vc_model_tag_data"] = asdict(ModelTagData())

    elif full_file_path:
        if get_model_binary_file_extension() in full_file_path:
            from medcat.cdb import CDB
            from medcat.vocab import Vocab
                
            try:
                if "cdb" in full_file_path.lower():
                    data = CDB.load(full_file_path)
                elif "vocab" in full_file_path.lower():
                    data = Vocab.load(full_file_path)

            except Exception as exception:
                    logging.error("could not add vc_model_tag_data attribute to model data file")
                    logging.error(repr(exception))
                    logging.error(traceback.format_exc())
                    return False         
                    
        elif ignore_non_model_files:
            pass
        elif ".npy" in full_file_path:
            data = numpy.load(full_file_path)
        else:
            with open(full_file_path, "rb") as f:
                data = dill.load(f)
    return data

def get_tag_str_model_name_and_version(full_model_tag_name: str, delimiter: str = '-'):
    """
        Splits the tag name to get the model name without the prefix (if prx`esent)
        :returns: model_name, version_number
    """
    split_name_and_version = full_model_tag_name.split(delimiter)
    if len(split_name_and_version) >= 3:
        model_name = split_name_and_version[1]
    else:
        model_name = split_name_and_version[0]
    version = "1.0"
    if len(split_name_and_version) > 1:
        version = split_name_and_version[-1]
    return model_name, version

def prompt_statement(prompt_text, default_answer="yes"):
    valid_answers = {"yes": True, "no": False, "y": True, "n": False}
    exit_answer = ["exit", "cancel", "abort"]
    
    while True:
        print(prompt_text)
        print(" \t (Yes/No or Y/N), type exit/cancel/abort, or press CTRL+C to ABORT, all choices are case insensitive!")

        choice = input().lower()
 
        if choice in valid_answers.keys():
            return valid_answers[choice]
        else:
            print("Invalid answer, please try again...")
            if choice in exit_answer:
                sys.exit()

def create_model_folder(full_model_tag_name: str):
    try:
        model_dir_path = os.path.join(get_local_model_storage_path(), full_model_tag_name)
        if not os.path.isdir(model_dir_path):
            os.makedirs(model_dir_path)
        return True
        
    except Exception as exception:
        logging.info("Could not create model folder " + full_model_tag_name + ".")
        logging.info(repr(exception))

def get_downloaded_local_model_folder(full_model_tag_name: str):
    r'''
        Checks if folder for a specified model tag that has been downloaded exists 
        and it is a GIT repository (path checked is within the default local storage path).
        Returns False if either of conditions fail otherwise returns 
        the full path to the folder if conditions are true.

        Args:
            full_model_tag_name(`str`):
                Model name, e.g: sample_model_tag-1.0 
    '''
    try:
        full_model_path = os.path.join(get_local_model_storage_path(), full_model_tag_name)
        
        if os.path.exists(full_model_path) and os.path.isdir(full_model_path):
            if is_dir_git_repository(full_model_path):
                return full_model_path
        else:
            return False

    except Exception as exception:
        logging.error("Could not find model folder " + full_model_tag_name + ".")
        logging.error(repr(exception))
        return False

def get_local_model_storage_path(storage_path=os.path.join(os.path.expanduser("~"), ".cache", "medcat"), models_dir="models"):
    
    medcat_model_installation_dir_path = os.path.join(storage_path, models_dir)

    try:
        if not os.path.exists(storage_path):
            os.mkdir(storage_path)
        if not os.path.exists(medcat_model_installation_dir_path):
            os.mkdir(medcat_model_installation_dir_path)

    except OSError as os_exception:
        logging.error("Could not create MedCAT model storage folder: " + medcat_model_installation_dir_path)
        logging.error(repr(os_exception))

    if os.access(medcat_model_installation_dir_path, os.R_OK | os.X_OK | os.W_OK):
        return medcat_model_installation_dir_path

    return ""
    
def copy_model_files_to_folder(source_folder, dest_folder):

    root, subdirs, files = next(os.walk(source_folder))

    print("files : ", files)
    for file_name in files:
        if file_name in get_permitted_push_file_list():
            logging.info("Copying file : " + file_name + " to " + dest_folder)
            shutil.copy2(os.path.join(source_folder, file_name), dest_folder)
        else:
            logging.info("Discarding " + file_name + " as it is not in the permitted model file pushing convention...")

def create_new_base_repository(repo_folder_path, git_repo_url, remote_name="origin", branch="master", checkout_full_tag_name=""):
    """
        Creates a base repository for a NEW base model release. 
        The base repo always points to the HEAD commit of the git history, not to a tag/release commit.

        :param checkout_full_tag_name : should be used in the case of creating/updating from already existing model release/tag 
    """

    try:
        subprocess.run(["git", "init"], cwd=repo_folder_path)
        subprocess.run(["git", "remote", "add", remote_name, git_repo_url], cwd=repo_folder_path)
        subprocess.run(["git", "fetch", "--tags", "--force"], cwd=repo_folder_path)
        
        if checkout_full_tag_name != "":
             subprocess.run(["git", "checkout", "tags/" + checkout_full_tag_name, "-b" , branch], cwd=repo_folder_path)

        subprocess.run(["git", "pull", remote_name, branch], cwd=repo_folder_path)
        return True

    except Exception as exception:
        logging.error("Error creating base model repository: " + repr(exception))
        return False

def is_dir_git_repository(path):
    try:
        repo = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError as exception:
        logging.error("Folder:" + path + " is not a git repository. Description:" + repr(exception))
        return False

def file_is_hidden(path):
    if os.name == "nt":
        attribute = win32api.GetFileAttributes(path)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return path.startswith('.')

def force_delete_path(path):
    if os.name == "nt":
        win32api.SetFileAttributes(path, win32con.FILE_ATTRIBUTE_NORMAL)
    for root, dirs, files in os.walk(path):  
        for dir in dirs:
            os.chmod(os.path.join(root, dir), stat.S_IRWXU)
        for file in files:
            os.chmod(os.path.join(root, file), stat.S_IRWXU)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.isfile(path):
        os.remove(path)

def get_medcat_package_version():
    import pkg_resources 
    version = pkg_resources.require("medcat")[0].version
    return str(version)

def sanitize_input(input_string):
    input_string = re.compile('-').sub('_', input_string).strip()
    input_string = re.compile("[^a-zA-Z_]").sub('', input_string).strip("_")
    return input_string

def is_input_valid(input_string): 
    return len(re.findall("[^a-zA-Z_]", input_string)) == 0
    
def get_model_binary_file_extension():
    return ".dat"

def get_permitted_push_file_list():
    return ["cdb.dat", "vocab.dat", "modelcard.md", "modelcard.json", "MedCAT_Export.json" , "embeddings.npy"]

def get_permitted_versioned_files():
    return ["cdb.dat", "vocab.dat", "MedCAT_Export.json"]

def dict_diff(dict_1, dict_2):
    diff_1, diff_2 = {}, {}

    dict_1_key_diff = set(dict_1.keys()) - set(dict_2.keys())
    dict_2_key_diff = set(dict_2.keys()) - set(dict_1.keys()) 

    if len(dict_1_key_diff) > 0:
        for k1 in dict_1_key_diff:
            diff_2[k1] = []

    if len(dict_2_key_diff) > 0:
        for k2 in dict_2_key_diff:
            diff_1[k2] = []
    
    for k1, v1 in dict_1.items():
        for k2, v2 in dict_2.items():
            diffv1 = set(v1) - set(v2)

            if len(diffv1) > 0:
                diff_1[k1]= diffv1
            
            diffv2 = set(v2) - set(v1)
            if len(diffv2) > 0:
                diff_2[k2] = diffv2
            

    return diff_1, diff_2

def make_orderer():
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        return [1, -1][order[a] < order[b]]

    return ordered, compare