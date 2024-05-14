# TODO: REMOVE BELOW
import unittest
from tests.stats.test_kfold import debug_print_test_names


@debug_print_test_names
class DebugTestCase(unittest.TestCase):
    pass


# Override unittest.TestCase with DebugTestCase
unittest.TestCase = DebugTestCase
# TODO: REMOVE ABOVE
