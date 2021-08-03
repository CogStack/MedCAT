name="medcat.cli"

from .package import package
from .config import config
from .download import download
from .listmodels import listmodels
from .modeltagdata import ModelTagData
from .system_utils import *

import logging
logging.basicConfig(level=logging.INFO)