from unittest import TestCase

from medcat.utils.regression import utils


class PartialSubstituationTests(TestCase):
    TEXT1 = "This [PH1] has one placeholder"
    PH1 = "PH1"
    REPLACEMENT1 = "<REPLACE1>"

    def test_fails_with_1_ph(self):
        with self.assertRaises(utils.IncompatiblePhraseException):
            utils.partial_substitute(self.TEXT1, self.PH1, self.REPLACEMENT1, 0)

    TEXT2 = "This [PH1] has [PH1] multiple (2) placeholders"

    def assert_is_correct_for_regr(self, text: str, placeholder: str):
        # should leave a placeholder in
        self.assertIn(placeholder, text)
        # and only 1
        self.assertEqual(text.count(placeholder), 1)

    def assert_has_replaced_and_is_suitable(self, text: str, placeholder: str, replacement: str,
                                            repl_count: int):
        self.assert_is_correct_for_regr(text, placeholder)
        self.assertIn(replacement, text)
        self.assertEqual(text.count(replacement), repl_count)

    def test_works_with_2_ph_0th(self):
        text = utils.partial_substitute(self.TEXT2, self.PH1, self.REPLACEMENT1, 0)
        self.assert_has_replaced_and_is_suitable(text, self.PH1, self.REPLACEMENT1, 1)

    def test_works_with_2_ph_1st(self):
        text = utils.partial_substitute(self.TEXT2, self.PH1, self.REPLACEMENT1, 1)
        self.assert_has_replaced_and_is_suitable(text, self.PH1, self.REPLACEMENT1, 1)

    def test_fails_if_too_high_a_change_nr(self):
        with self.assertRaises(utils.IncompatiblePhraseException):
            utils.partial_substitute(self.TEXT1, self.PH1, self.REPLACEMENT1, 2)

    TEXT3 = "No [PH1] is [PH1] safe [PH1] eh"

    def test_work_with_3_ph(self):
        for nr in range(self.TEXT3.count(self.PH1)):
            with self.subTest(f"Placeholder #{nr}"):
                text = utils.partial_substitute(self.TEXT3, self.PH1, self.REPLACEMENT1, nr)
                self.assert_has_replaced_and_is_suitable(text, self.PH1, self.REPLACEMENT1, 2)

    def test_all_possibilities_are_similar(self):
        texts = [utils.partial_substitute(self.TEXT3, self.PH1, self.REPLACEMENT1, nr)
                 for nr in range(self.TEXT3.count(self.PH1))]
        # they should all have the same length
        lengths = [len(t) for t in texts]
        self.assertTrue(all(cl == lengths[0] for cl in lengths))
        # they should all have the same character composition
        # i.e they should compose of the same exact characters
        char_compos = [set(t) for t in texts]
        self.assertTrue(all(cchars == char_compos[0] for cchars in char_compos))
        # and there should be the same amount for each as well
        char_counts = [{c: t.count(c) for c in char_compos[0]} for t in texts]
        self.assertTrue(all(cchars == char_counts[0] for cchars in char_counts))    
