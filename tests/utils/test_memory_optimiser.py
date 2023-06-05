from medcat.utils import memory_optimiser

import unittest


class DelegatingDictTests(unittest.TestCase):
    _dict = {'c1': [None, 0], 'c2': [1, None]}

    def setUp(self) -> None:
        self.del_dict1 = memory_optimiser.DelegatingDict(self._dict, 0, 2)
        self.del_dict2 = memory_optimiser.DelegatingDict(self._dict, 1, 2)
        self.delegators = [self.del_dict1, self.del_dict2]
        self.names = ['delegator 1', 'delegator 2']
        self.expected_lens = [len(
            [v[nr] for v in self._dict.values() if v[nr] is not None]
        ) for nr in range(len(self._dict[list(self._dict.keys())[0]]))]

    def test_delegating_dict_has_correct_keys(self):
        for delegator, exp_len, name in zip(self.delegators, self.expected_lens, self.names):
            with self.subTest(name):
                self.assertEqual(len(delegator.keys()), exp_len)

    def test_delegating_dict_has_same_number_of_keys_and_values(self):
        for delegator, exp_len, name in zip(self.delegators, self.expected_lens, self.names):
            with self.subTest(name):
                self.assertEqual(len(delegator.keys()), exp_len)
                self.assertEqual(len(delegator.values()), exp_len)

    def test_delegating_dict_has_same_number_of_items_and_iter_values(self):
        for delegator, exp_len, name in zip(self.delegators, self.expected_lens, self.names):
            with self.subTest(name):
                self.assertEqual(len(delegator.items()), exp_len)
                # __iter__ -> list -> len
                self.assertEqual(len(list(delegator)), exp_len)

    def test_delegator_do_not_have_None_values(self):
        for delegator, name in zip(self.delegators, self.names):
            for key, val in delegator.items():
                with self.subTest(f"{name}: {key}"):
                    self.assertIsNotNone(val)

    def test_delegator_keys_in_original(self):
        for delegator, name in zip(self.delegators, self.names):
            for key in delegator.keys():
                with self.subTest(f"{name}: {key}"):
                    self.assertIn(key, self._dict)

    def test_delegator_keys_in_container(self):
        for delegator, name in zip(self.delegators, self.names):
            for key in delegator.keys():
                with self.subTest(f"{name}: {key}"):
                    self.assertIn(key, delegator)

    def test_delegator_get_gets_key(self, def_value='#DEFAULT#'):
        for delegator, name in zip(self.delegators, self.names):
            for key in delegator.keys():
                with self.subTest(f"{name}: {key}"):
                    val = delegator.get(key, def_value)
                    self.assertIsNot(val, def_value)

    def test_delegator_get_defaults_non_existant_key(self, def_value='#DEFAULT#'):
        for delegator, name in zip(self.delegators, self.names):
            for key in self._dict.keys():
                if key in delegator:
                    continue
                with self.subTest(f"{name}: {key}"):
                    val = delegator.get(key, def_value)
                    self.assertIs(val, def_value)
