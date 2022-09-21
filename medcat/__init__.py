import os
from pkg_resources import get_distribution, DistributionNotFound

import logging

# Make sure the default behaviour is to _not_ log anything
# The idea is to let the user of the library decide where,
# what and how they want to log information from the library
logging.getLogger(__name__).addHandler(logging.NullHandler())

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
