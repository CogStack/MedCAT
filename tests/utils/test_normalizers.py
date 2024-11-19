import unittest

from medcat.utils import normalizers


class EditOrderTests(unittest.TestCase):
    WORD = "abc"
    EXMAPLE_EDITS_ORDER = [
            'abqc', 'rbc', 'obc', 'fbc', 'abyc',
            'azbc', 'ibc', 'xbc', 'apc', 'abcl',
            'abcr', 'abck', 'anc', 'abd', 'abkc',
            'iabc', 'tbc', 'cabc', 'abw', 'abp',
            'abe', 'akbc', 'apbc', 'hbc', 'ubc',
            'abic', 'babc', 'abcq', 'wabc', 'abtc',
            'aibc', 'yabc', 'asc', 'abrc', 'avbc',
            'abu', 'kabc', 'axc', 'fabc', 'nbc',
            'rabc', 'abec', 'abcu', 'gbc', 'amc',
            'abce', 'abdc', 'abcy', 'bbc', 'dbc',
            'abac', 'abvc', 'abuc', 'avc', 'abi',
            'abm', 'abjc', 'abcp', 'tabc', 'cbc',
            'uabc', 'abz', 'aby', 'qbc', 'abcf',
            'abpc', 'axbc', 'abk', 'gabc', 'abc',
            'mbc', 'aqbc', 'abci', 'oabc', 'qabc',
            'abf', 'vabc', 'abj', 'abbc', 'aubc',
            'acbc', 'abn', 'aebc', 'ebc', 'abfc',
            'dabc', 'abh', 'arc', 'aqc', 'albc',
            'aac', 'abcb', 'sabc', 'ybc', 'abcv',
            'absc', 'abca', 'labc', 'ajbc', 'kbc',
            'pabc', 'abcc', 'afbc', 'sbc', 'abl',
            'awc', 'ahbc', 'abco', 'anbc', 'abo',
            'abg', 'abcn', 'awbc', 'adc', 'ahc',
            'habc', 'abb', 'vbc', 'aboc', 'abq',
            'acc', 'agc', 'abcx', 'nabc', 'abwc',
            'lbc', 'abcm', 'afc', 'ab', 'atc',
            'aybc', 'akc', 'abt', 'aic', 'jbc',
            'aec', 'zabc', 'agbc', 'abv', 'abnc',
            'abcj', 'pbc', 'abcg', 'bac', 'abr',
            'aobc', 'abcd', 'alc', 'aoc', 'ajc',
            'abx', 'arbc', 'ayc', 'aba', 'abcw',
            'eabc', 'abcs', 'abhc', 'adbc', 'abgc',
            'asbc', 'acb', 'abs', 'aabc', 'abzc',
            'abxc', 'atbc', 'ambc', 'jabc', 'bc',
            'wbc', 'abcz', 'ablc', 'ac', 'azc',
            'abct', 'abmc', 'zbc', 'abch', 'auc',
            'xabc', 'mabc'
        ]

    # NOTE: The there is a chance that this test fails. But it should be 2 in 182!
    #       (since I'm checking against 2 different orders - the one captured above
    #       and the alphabetically ordered version calculated on the fly). This is
    #       essentially 0 and _should_ never happen.
    def test_order_not_guaranteed1(self):
        all_edits = list(normalizers.get_all_edits_n(self.WORD, use_diacritics=False, n=1, return_ordered=False))
        self.assertNotEqual(all_edits, self.EXMAPLE_EDITS_ORDER)
        self.assertNotEqual(all_edits, sorted(all_edits))

    def test_ordered_within_same_run1(self):
        all_edits1 = list(normalizers.get_all_edits_n(self.WORD, use_diacritics=False, n=1, return_ordered=False))
        all_edits2 = list(normalizers.get_all_edits_n(self.WORD, use_diacritics=False, n=1, return_ordered=False))
        self.assertEqual(all_edits1, all_edits2)

    def test_all_items_same_now1(self):
        all_edits = list(normalizers.get_all_edits_n(self.WORD, use_diacritics=False, n=1, return_ordered=False))
        for got_now in all_edits:
            with self.subTest(got_now):
                self.assertIn(got_now, self.EXMAPLE_EDITS_ORDER)

    def test_can_guarantee_order1(self):
        all_edits1 = list(normalizers.get_all_edits_n(self.WORD, use_diacritics=False, n=1, return_ordered=True))
        ordered = sorted(all_edits1)
        self.assertEqual(all_edits1, ordered)
