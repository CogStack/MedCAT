import fire
from medcat.cli.download import get_all_available_model_tags
from medcat.cli.config import get_auth_environment_vars

def listmodels():
    print("Checking " + get_auth_environment_vars()["git_repo_url"] + " for releases...")
    print("The following model tags are available for download : " + str(get_all_available_model_tags()))

if __name__ == '__main__':
    fire.Fire(listmodels)