from medcat.utils.saving import coding

import json

import unittest


class SetEncodeTests(unittest.TestCase):
    string2sets_dict1 = {'s1': set(['v1', 'v2', 'v3']),
                         's2': set(['u1', 'u2', 'u3'])}
    string2sets_dict2 = {'p1': set([1, 2, 3]),
                         'p2': set([3, 4, 5])}

    def serialise(self, d: dict) -> str:
        return json.dumps(d, cls=coding.CustomDelegatingEncoder.def_inst)

    def _helper_serialises(self, d: dict):
        s = self.serialise(d)
        self.assertIsInstance(s, str)

    def test_sets_of_strings_serialise(self):
        self._helper_serialises(self.string2sets_dict1)

    def test_sets_of_ints_serialise(self):
        self._helper_serialises(self.string2sets_dict2)

    def _helper_keys_in_json(self, d: dict):
        s = self.serialise(d)
        for k in d.keys():
            with self.subTest(k):
                self.assertIn(str(k), s)

    def test_sos_keys_in_json(self):
        self._helper_keys_in_json(self.string2sets_dict1)

    def test_soi_keys_in_json(self):
        self._helper_keys_in_json(self.string2sets_dict2)

    def _helper_values_in_json(self, d: dict):
        s = self.serialise(d)
        for key, v in d.items():
            for nr, el in enumerate(v):
                with self.subTest(f"Key: {key}; Element {nr}"):
                    self.assertIn(str(el), s)

    def test_sos_values_in_json(self):
        self._helper_values_in_json(self.string2sets_dict1)

    def test_soi_values_in_json(self):
        self._helper_values_in_json(self.string2sets_dict2)


class SetDecodeTests(unittest.TestCase):

    def deserialise(self, s: str) -> dict:
        return json.loads(s, object_hook=coding.default_hook)

    def setUp(self) -> None:
        self.encoder = SetEncodeTests()
        self.encoded1 = self.encoder.serialise(self.encoder.string2sets_dict1)
        self.encoded2 = self.encoder.serialise(self.encoder.string2sets_dict2)

    def test_sos_decodes(self):
        d = self.deserialise(self.encoded1)
        self.assertIsInstance(d, dict)

    def test_soi_decodes(self):
        d = self.deserialise(self.encoded2)
        self.assertIsInstance(d, dict)

    def test_sos_decodes_to_identical(self):
        d = self.deserialise(self.encoded1)
        self.assertEqual(d, self.encoder.string2sets_dict1)

    def test_soi_decodes_to_identical(self):
        d = self.deserialise(self.encoded2)
        self.assertEqual(d, self.encoder.string2sets_dict2)
