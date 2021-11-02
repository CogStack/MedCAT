import sys
import fire
import os
import json
import logging
import medcat
import subprocess

from medcat.cli.system_utils import is_input_valid, prompt_statement, sanitize_input


env_var_field_mapping = {
                         "username": "MEDCAT_GIT_USERNAME",
                         "git_auth_token" : "MEDCAT_GIT_AUTH_TOKEN",
                         "git_repo_url" : "MEDCAT_GIT_REPO_URL",
                         "git_organisation_name" : "MEDCAT_GIT_ORGANISATION_NAME",
                         "storage_server_username" : "STORAGE_SERVER_USERNAME",
                         "storage_server_remote_name" : "STORAGE_SERVER_REMOTE_NAME",
                         "storage_server_password" : "STORAGE_SERVER_PASSWORD",
                         "storage_server_model_repo_path" : "STORAGE_SERVER_MODEL_REPO_PATH"
                        }

test_mode_settings = ['storage_server_username', 'storage_server_password', 'storage_server_model_repo_path', 'storage_server_remote_name']

def config():
    """
        Reads the user input and sets the repository configurations.
    """
    config_data = {}
    
    for k,v in env_var_field_mapping.items():
        while True:
            input_val = input("Please input your " + k + " (" + v + ") : ")
            if k in test_mode_settings:
                logging.info("the value is optional, leave this value empty as it should be used for TESTING purposes only, the storage password and model repo path & name are stored globally for testing")

            config_data[v] = input_val.strip()
            if k == "git_organisation_name":
                if input_val.strip() == "":
                    config_data[v] = get_git_user_project(config_data[env_var_field_mapping["git_repo_url"]]).split("/")[0]
                    logging.info(" " + env_var_field_mapping[k] + " not set, inferring ORGANISATION name from the git repo : " + "\033[1m" +  config_data[env_var_field_mapping["git_repo_url"]] + "\033[0m" +
                    " \n the organisation name will be : " + config_data[v])

                if not is_input_valid(config_data[v]):
                   logging.info(config_data[v] + " is not a valid input as it contains restricted characters, please use only letters and underscores")
                   sanitized_input =  sanitize_input(config_data[v])
                   if prompt_statement("The organisation name has been sanitized, and it will look as follows: " + sanitized_input + " \n Are you satisfied with this name ?"):
                       config_data[v] = sanitized_input
                       break
                else:
                    break

            if k in test_mode_settings or input_val.strip() != "":
                break

    generate_medcat_config_file(config_data)

def set_git_global_git_credentials():
    """
        Set the git credentials according to the settings in the config.
        This should be used for testing and automated systems only.
    """
    subprocess.run(["git", "config", "--global", "--unset-all", "credential.helper"])

    if os.name == "nt":
        subprocess.run(["git", "config", "--global", "credential.helper", "wincred", "'cache --timeout=1800'"])
    else:
        subprocess.run(["git", "config", "--global", "credential.helper", "cache", "'cache --timeout=1800'"])

    subprocess.run(["git", "config", "--global", "user.name", get_auth_environment_vars()["username"]])
    subprocess.run(["git", "config", "--global", "user.password", get_auth_environment_vars()["git_auth_token"]])

def get_auth_environment_vars():
    """
        :returns: a dict with the github username, auth token, repo url and organisation name
    """
    auth_vars = { 
                 "username" : os.getenv(env_var_field_mapping["username"], ""), 
                 "git_auth_token": os.getenv(env_var_field_mapping["git_auth_token"], ""), 
                 "git_repo_url":  os.getenv(env_var_field_mapping["git_repo_url"], ""), 
                 "git_organisation_name" : os.getenv(env_var_field_mapping["git_organisation_name"], ""),
                 "storage_server_username" : os.getenv(env_var_field_mapping["storage_server_username"], ""),
                 "storage_server_remote_name" :  os.getenv(env_var_field_mapping["storage_server_remote_name"], ""),
                 "storage_server_password" : os.getenv(env_var_field_mapping["storage_server_password"], ""),
                 "storage_server_model_repo_path" : os.getenv(env_var_field_mapping["storage_server_model_repo_path"], "")
                }
    try:
        env_medcat_config_file = get_medcat_config_settings()
      
        for k,v in auth_vars.items():
            if k not in test_mode_settings and v.strip() == "" and env_var_field_mapping[k] in env_medcat_config_file.keys() and env_medcat_config_file[env_var_field_mapping[k]] != "":
                auth_vars[k] = env_medcat_config_file[env_var_field_mapping[k]]
            if auth_vars[k].strip() == "" and k not in test_mode_settings:
                logging.error("Please set your configuration settings by using the 'python3 -m medcat config' command or by exporting the global variable in your current session 'export " + env_var_field_mapping[k] + "=your_value' !")
                logging.error("CONFIG NOT SET for : " + k + " , from environment var : " + env_var_field_mapping[k])
                sys.exit()
                
        return auth_vars

    except Exception as exception:
        logging.error(repr(exception))
        return False

def generate_medcat_config_file(config_settings={}, config_dirname="config", config_file="env_version_control.json"):
    config_path = os.path.join(os.path.dirname(medcat.__file__), config_dirname)
    try:
        if os.path.isdir(config_path) is False:
            os.makedirs(config_path)  
    except Exception as exception:
        logging.error("Could not create MedCAT config folder " + str(config_path))
        logging.error(repr(exception))

    with open(os.path.join(config_path, config_file), "w") as f:
        json.dump(config_settings, f)
        logging.info("Config file saved in : " + str(os.path.join(config_path, config_file)))

def get_medcat_config_settings(config_dirname="config", config_file="env_version_control.json"):
    config_file_contents = {}
    config_file_full_path = os.path.join(os.path.dirname(medcat.__file__), config_dirname, config_file)
    if os.path.isfile(config_file_full_path):
        with open(config_file_full_path, "r") as f:
            config_file_contents = json.load(f)
    return config_file_contents    

def get_git_user_project(url=""):
    """
        :return: user/repo_name from git url, e.g: https://github.com/user/repo_name -> user/repo_name
    """
    if url.strip() == "":
        env_git_repo_url = get_auth_environment_vars()["git_repo_url"]
    else:
        env_git_repo_url = url
    user_repo_and_project = '/'.join(env_git_repo_url.split('.git')[0].split('/')[-2:])
    return user_repo_and_project

def get_git_default_headers():
    env_git_auth_token = get_auth_environment_vars()["git_auth_token"]
    headers = {"Accept" : "application/vnd.github.v3+json", "Authorization": "token " + env_git_auth_token} 
    return headers

def get_git_download_headers():
    headers = get_git_default_headers()
    headers["Accept"] =  "application/octet-stream"
    return headers

def get_git_upload_headers():
    headers = get_git_default_headers()
    headers = {**headers, "Content-Type": "application/octet-stream"}
    return headers 

def get_git_api_request_url():
    return "https://api.github.com/repos/" + get_git_user_project() + "/"

def get_git_api_upload_url():
    return "https://uploads.github.com/repos/" + get_git_user_project() + "/"

if __name__ == '__main__':
  fire.Fire(config)