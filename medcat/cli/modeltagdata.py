from dataclasses import dataclass

@dataclass
class ModelTagData:
    organisation_name: str = ""
    model_name: str = ""
    parent_model_name: str = ""
    version: str = ""
    commit_hash: str = ""
    git_repo: str = ""
    parent_model_tag: str = ""
    storage_location: str = ""
    medcat_version: str = ""
    authors : str = ""