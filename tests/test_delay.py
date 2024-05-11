import time
import unittest


class DelayTest(unittest.TestCase):
    def test_delay(self):
        print("Sleeping...")
        time.sleep(8*60) # 8 minutes
        print("Done sleeping")
