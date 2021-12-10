import unittest
from medcat.utils.decorators import check_positive


class DecoratorsTest(unittest.TestCase):

    def test_check_positive(self):
        @check_positive
        def func(value1, value2=None):
            return value1, value2

        with self.assertRaises(ValueError) as e:
            func(-1)
        self.assertEqual("Argument at position 0 is not a positive integer", str(e.exception))

        with self.assertRaises(ValueError) as e:
            func(1, -1)
        self.assertEqual("Argument at position 1 is not a positive integer", str(e.exception))
