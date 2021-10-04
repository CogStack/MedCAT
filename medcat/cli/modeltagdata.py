from dataclasses import dataclass, field
from os import times
from typing import List

import datetime
from datetime import timezone

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
    authors : List[str] = field(default_factory=list)
    timestamp : str = datetime.datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')