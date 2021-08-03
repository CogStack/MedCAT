import fire
import requests
import os
import sys
import subprocess
import logging
from .system_utils import *
from medcat.cli.config import get_auth_environment_vars, get_git_api_request_url, get_git_default_headers, get_git_download_headers

def get_matching_version(full_model_tag_name):

    request_url = get_git_api_request_url() + 'releases/tags/' + full_model_tag_name
    response = requests.get(request_url, headers=get_git_default_headers())

    result = {'request_success': False, 'credentials_correct': True ,'response_message': '', 'tag_asset_id': ''}
   
    if response.status_code == 200:
        result['request_success'] = True
        for asset in response.json()['assets']:
            if asset['name'] == str(full_model_tag_name + get_asset_file_extension()):   
                result['tag_asset_id'] = str(asset['id'])
    if response.status_code == 404:
        result['response_message'] = response.json()['message']
    if response.status_code == 401:
        result['response_message'] = response.json()['message']
        result['credentials_correct'] = False

    return result

def download_asset(full_model_tag_name, asset_id):

    downloaded_tag_bundle_file = requests.get(get_git_api_request_url() + "releases/assets/" + asset_id, headers=get_git_download_headers())

    if downloaded_tag_bundle_file.status_code == 200:
        model_asset_file_and_folder_location = get_local_model_storage_path()

        if model_asset_file_and_folder_location != "":
            model_asset_file = os.path.join(model_asset_file_and_folder_location, full_model_tag_name) + get_asset_file_extension()
            with open(model_asset_file, 'wb') as f:
               f.write(downloaded_tag_bundle_file.content)
               print("Downloaded model package file to : ", model_asset_file)

            return True
    else:
        logging.error("Could not download model package file : " + str(downloaded_tag_bundle_file.status_code) + ", " + downloaded_tag_bundle_file.text)

    return False

def get_all_available_model_tags():
    list_tags_req = requests.get(url=get_git_api_request_url() + "tags", headers=get_git_default_headers())
    model_tag_names = []

    if list_tags_req.status_code == 200:
        for tag in list_tags_req.json():
            model_tag_names.append(tag["name"])
    else:
        logging.error("Failed to fetch list of all releases available: " + str(list_tags_req.status_code) + " " + list_tags_req.text)

    return model_tag_names

def unpack_asset(full_model_tag_name, git_repo_url, remote_name="origin", branch="master"):
    try:
        model_storage_path = get_local_model_storage_path()
        if model_storage_path != "":
            model_asset_dir_location = os.path.join(model_storage_path, full_model_tag_name)
            model_asset_bundle_file = model_asset_dir_location + get_asset_file_extension()

            if os.path.exists(model_asset_dir_location) and os.path.isdir(model_asset_dir_location):
                print("Found previous installation of model: " + full_model_tag_name, " , in path: ", model_asset_dir_location)
                if prompt_statement("Should this installation be deleted and reinstalled ?"):
                    force_delete_path(model_asset_dir_location)
                else:
                    sys.exit()  
                    
            subprocess.run(["git", "clone", model_asset_bundle_file], cwd=model_storage_path)  

            if is_dir_git_repository(model_asset_dir_location):
                subprocess.run(["git", "remote", "remove", remote_name], cwd=model_asset_dir_location)
                subprocess.run(["git", "remote", "add", remote_name, git_repo_url], cwd=model_asset_dir_location)

            if os.path.isfile(model_asset_bundle_file):
                os.remove(model_asset_bundle_file)

            subprocess.run([sys.executable, "-m", "dvc", "pull"], cwd=model_asset_dir_location)
            
    except Exception as exception:
        logging.error("Error unpacking model file asset : " + repr(exception))

def download(full_model_tag_name):

    git_repo_url = get_auth_environment_vars()["git_repo_url"]
    request_url = get_git_api_request_url()

    # Try to get exact match:
    result = get_matching_version(full_model_tag_name)

    if result["request_success"]:
        print("Found release ", full_model_tag_name, ". Downloading...")
        if result["tag_asset_id"]:
          download_asset(full_model_tag_name, asset_id=result["tag_asset_id"])
          unpack_asset(full_model_tag_name, git_repo_url)
        else:
            print("Release tag " + full_model_tag_name + " asset id not found, please retry...")
    else:
        available_model_release_tag_names = get_all_available_model_tags()
       
        if available_model_release_tag_names:
            matching_tag_names = []
            for tag_name in available_model_release_tag_names:
                if full_model_tag_name in tag_name:
                    matching_tag_names.append(tag_name)
        
            if matching_tag_names:
                print("Found the following tags with a similar name:")
                print(matching_tag_names)
                while True:
                    model_choice = input("Please input the model version you would like to download: \n")
                    if model_choice in matching_tag_names:
                        result = get_matching_version(model_choice)
                        print("Found release ", model_choice, ". Downloading...")
                        download_asset(model_choice, asset_id=result["tag_asset_id"])
                        unpack_asset(model_choice, git_repo_url)
                        break
            else:
                print("No release tags found with the given name or containing a similar name, however, the following releases are available:")
                print(available_model_release_tag_names)
                while True:
                    model_choice = input("Please input the model version you would like to download: \n")
                    if model_choice in available_model_release_tag_names:
                        result = get_matching_version(model_choice)
                        print("Found release ", model_choice, ". Downloading...")
                        download_asset(model_choice, asset_id=result["tag_asset_id"])
                        unpack_asset(model_choice, git_repo_url)
                        break
        else:
            print("No model release tags found on repository: " + request_url)
            print("Make sure you have configured MedCAT repository settings via the configure command.")
            sys.exit()
    return True

def get_asset_file_extension():
    return ".bundle"
                
if __name__ == '__main__':
  fire.Fire(download)