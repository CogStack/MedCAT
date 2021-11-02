
from dataclasses import dataclass, field
from os import times
from typing import List
from datetime import datetime, timezone

from medcat.cli.global_settings import DEFEAULT_DATETIME_FORMAT

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
    timestamp : str = field(default_factory=lambda: datetime.now(tz=timezone.utc).strftime(DEFEAULT_DATETIME_FORMAT))