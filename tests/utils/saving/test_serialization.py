import os
import shutil

import tempfile
import unittest

from medcat.cdb import CDB

from medcat.utils.saving.serializer import JsonSetSerializer, CDBSerializer, SPECIALITY_NAMES


class JSONSerialoizationTests(unittest.TestCase):
    folder = os.path.join('temp', 'JSONSerialoizationTests')

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.folder)
        return super().tearDown()

    def test_json_serializes_set_round_truip(self):
        d = {'val': {'a', 'b', 'c'}}
        ser = JsonSetSerializer(self.folder, 'test_json.json')
        ser.write(d)
        back = ser.read()
        self.assertEqual(d, back)


class CDBSerializationTests(unittest.TestCase):
    test_file = tempfile.NamedTemporaryFile()

    def setUp(self) -> None:
        self.cdb = CDB.load(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "..", "..", "..", "examples", "cdb.dat"))
        self.ser = CDBSerializer(self.test_file.name)

    def test_round_trip(self):
        self.ser.serialize(self.cdb, overwrite=True)
        cdb = self.ser.deserialize(CDB)
        for name in SPECIALITY_NAMES:
            with self.subTest(name):
                orig = getattr(self.cdb, name)
                now = getattr(cdb, name)
                if not orig == now:
                    print(name, '\nORIG\n', orig, '\n vs NOW\n', now)
                assert orig == now
