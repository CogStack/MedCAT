import os
from pkg_resources import get_distribution, DistributionNotFound

import logging

# Make sure the default behaviour is to _not_ log anything
# The idea is to let the user of the library decide where,
# what and how they want to log information from the library
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def add_default_log_handlers(log: logging.Logger = logger, target_file: str = 'medcat.log') -> None:
    """Add default log handlers to the specified logger.
    This method will add a file handler that will write
    into `target_file` (defaults to 'medcat.log') and
    a console handler that outputs to the console.

    Args:
        log (logging.Logger): The logger to add the handlers to. Defaults to logger.
        target_file (str): The target file for file handler. Defaults to 'medcat.log'.
    """
    # If we do not have any non-null handlers add them
    if len(log.handlers) == 0 or (len(log.handlers) == 1 and isinstance(log.handlers[0], logging.NullHandler)):
        # create a file handler
        fh = logging.FileHandler(target_file)
        # create console handler
        ch = logging.StreamHandler()

        # add the handlers to the logger
        log.addHandler(fh)
        log.addHandler(ch)


name = 'medcat'

try:
    distribution = get_distribution(name)
    distribution_location = os.path.normcase(distribution.location)
    init_location = os.path.normcase(__file__)
    if not init_location.startswith(os.path.join(distribution_location, name)):
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = f'Please install the package: {name}'
else:
    __version__ = distribution.version
