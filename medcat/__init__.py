import os
from pkg_resources import get_distribution, DistributionNotFound

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
