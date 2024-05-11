import time
import unittest


class DelayTest(unittest.TestCase):
    def test_delay(self):
        print("Sleeping...")
        time.sleep(3*60) # 3 minutes
        print("Done sleeping")
