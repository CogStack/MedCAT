import logging
name="medcat.cli"

from .package import package # noqa
from .config import config # noqa
from .download import download # noqa
from .listmodels import listmodels # noqa
from .modeltagdata import ModelTagData # noqa
from .system_utils import * # noqa

logging.basicConfig(level=logging.INFO)
