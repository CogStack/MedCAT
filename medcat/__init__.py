name = 'medcat'

# Hacky patch to the built-in copy module coz otherwise, thinc.config.Config.copy will fail on Python <= 3.6.
# (fixed in python 3.7 https://docs.python.org/3/whatsnew/3.7.html#re)
import sys # noqa

if sys.version_info.major == 3 and sys.version_info.minor <= 6:
    import copy
    import re

    copy._deepcopy_dispatch[type(re.compile(''))] = lambda r, _: r
