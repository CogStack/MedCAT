import os

from medcat.config import UsageMonitor as UsageMonitorConfig
from medcat.utils import usage_monitoring

import tempfile

from unittest import TestCase


class UsageMonitorBaseTests(TestCase):
    MODEL_HASH = "MODEL_HASH"
    BATCH_SIZE = 10
    ALL_DATA = [
        (10, 10, 2), (100, 90, 4), (110, 110, 0)
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = UsageMonitorConfig(enabled=True,
                                        batch_size=cls.BATCH_SIZE)

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.config.log_folder = self._temp_dir.name
        self.monitor = usage_monitoring.UsageMonitor(self.MODEL_HASH,
                                                     self.config)
        for data in self.ALL_DATA:
            self.monitor.log_inference(*data)

    def _get_saved_lines(self) -> list:
        if not os.path.exists(self.monitor.log_file):
            return []
        with open(self.monitor.log_file) as f:
            return f.readlines()

    def tearDown(self) -> None:
        self.monitor.log_buffer.clear()
        self._temp_dir.cleanup()


class UsageMonitorInBufferTests(UsageMonitorBaseTests):
    BATCH_SIZE = 100

    def test_nothing_in_file(self):
        self.assertFalse(self._get_saved_lines())

    def test_all_in_buffer(self):
        lines = self.monitor.log_buffer
        self.assertEqual(len(lines), len(self.ALL_DATA))
        for data_nr, (data, line) in enumerate(zip(self.ALL_DATA, lines)):
            for sub_nr, nr in enumerate(data):
                with self.subTest(f"{data_nr}-{sub_nr} ({nr})"):
                    self.assertIn(str(nr), line)


class UsageMonitorInFileTests(UsageMonitorBaseTests):
    BATCH_SIZE = 1

    def test_nothing_in_buffer(self):
        self.assertFalse(self.monitor.log_buffer)

    def test_all_in_file(self):
        lines = self._get_saved_lines()
        self.assertEqual(len(lines), len(self.ALL_DATA))
        for data_nr, (data, line) in enumerate(zip(self.ALL_DATA, lines)):
            for sub_nr, nr in enumerate(data):
                with self.subTest(f"{data_nr}-{sub_nr} ({nr})"):
                    self.assertIn(str(nr), line)


class InterMediateUsageMonitorTests(UsageMonitorBaseTests):
    BATCH_SIZE = 2

    def setUp(self) -> None:
        super().setUp()
        total_items = len(self.ALL_DATA)
        self.expected_in_buffer = total_items % self.BATCH_SIZE
        self.expected_in_file = total_items - self.expected_in_buffer

    def test_some_in_buffer(self):
        self.assertTrue(self.monitor.log_buffer)
        self.assertEqual(len(self.monitor.log_buffer), self.expected_in_buffer)

    def test_some_in_file(self):
        lines = self._get_saved_lines()
        self.assertTrue(lines)
        self.assertEqual(len(lines), self.expected_in_file)
