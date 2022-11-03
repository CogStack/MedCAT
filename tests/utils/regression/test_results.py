
import unittest

from medcat.utils.regression.targeting import TranslationLayer
from medcat.utils.regression.results import FailReason


class TestFailReason(unittest.TestCase):
    cui2names = {
        'cui1': set(['name-cui1-1', 'name-cui1-2']),
        'cui2': set(['name-cui2-1', 'name-cui2-2']),
        'cui3': set(['name-cui3-1', 'name-cui3-2', 'name-cui3-3']),
        'cui4': set(['name-cui4-1', ]),
    }
    # only works if one name corresponds to one CUI
    name2cuis = dict([(name, set([cui]))
                     for cui, names in cui2names.items() for name in names])
    cui2type_ids = {
        'cui1': set(['T1', ]),
        'cui2': set(['T1', ]),
        'cui3': set(['T2', ]),
        'cui4': set(['T4', ])
    }
    cui2children = {}  # none for now
    tl = TranslationLayer(cui2names, name2cuis, cui2type_ids, cui2children)

    def test_cui_not_found(self, cui='cui-100', name='random n4m3'):
        fr = FailReason.get_reason_for(cui, name, {}, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_CUI_NOT_FOUND)

    def test_cui_name_found(self, cui='cui1', name='random n4m3-not-there'):
        fr = FailReason.get_reason_for(cui, name, {}, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_NAME_NOT_FOUND)


class TestFailReasonWithResultAndChildren(TestFailReason):
    res_w_cui1 = {'entities': {
        # cui1
        1: {'source_value': list(TestFailReason.cui2names['cui1'])[0], 'cui': 'cui1'},
    }}
    res_w_cui2 = {'entities': {
        # cui2
        1: {'source_value': list(TestFailReason.cui2names['cui2'])[0], 'cui': 'cui2'},
    }}
    res_w_both = {'entities': {
        # cui1
        1: {'source_value': list(TestFailReason.cui2names['cui1'])[0], 'cui': 'cui1'},
        # cui2
        2: {'source_value': list(TestFailReason.cui2names['cui2'])[0], 'cui': 'cui2'},
    }}
    cui2children = {'cui1': set(['cui2'])}
    tl = TranslationLayer(TestFailReason.cui2names, TestFailReason.name2cuis,
                          TestFailReason.cui2type_ids, cui2children)

    def test_found_child(self, cui='cui1', name='name-cui1-2'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_cui2, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_CUI_FOUND_CHILD)

    def test_found_parent(self, cui='cui2', name='name-cui2-1'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_cui1, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_CUI_FOUND_PARENT)


class TestFailReasonWithSpanningConcepts(unittest.TestCase):
    cui2names = {
        'cui1': set(['shallow', 'shallow2']),
        'cui1.1': set(['broader shallow', 'broader shallow2']),
        'cui1.1.1': set(['even broader shallow', 'even broader shallow2']),
        'cui2': set(['name-2', ]),
    }
    # only works if one name corresponds to one CUI
    name2cuis = dict([(name, set([cui]))
                     for cui, names in cui2names.items() for name in names])
    cui2type_ids = {
        'cui1': set(['T1', ]),
        'cui1.1': set(['T1', ]),
        'cui1.1.1': set(['T1', ])
    }
    cui2children = {}  # none for now
    tl = TranslationLayer(cui2names, name2cuis, cui2type_ids, cui2children)

    res_w_cui1 = {'entities': {
        # cui1
        1: {'source_value': list(cui2names['cui1'])[0], 'cui': 'cui1'},
    }}

    res_w_cui11 = {'entities': {
        # cui1.1
        1: {'source_value': list(cui2names['cui1.1'])[0], 'cui': 'cui1.1'},
    }}

    res_w_cui111 = {'entities': {
        # cui1.1.1
        1: {'source_value': list(cui2names['cui1.1.1'])[0], 'cui': 'cui1.1.1'},
    }}
    res_w_all = {'entities': dict([(nr, d['entities'][1]) for nr, d in enumerate([
        res_w_cui1, res_w_cui11, res_w_cui111])])}

    def test_gets_incorrect_span_big(self, cui='cui1', name='shallow'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_cui11, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_SPAN_BIG)

    def test_gets_incorrect_span_bigger(self, cui='cui1', name='shallow'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_cui111, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_SPAN_BIG)

    def test_gets_incorrect_span_small(self, cui='cui1.1', name='broader shallow'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_cui1, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_SPAN_SMALL)

    def test_gets_incorrect_span_smaller(self, cui='cui1.1.1', name='even broader shallow'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_cui1, self.tl)
        self.assertIs(fr, FailReason.INCORRECT_SPAN_SMALL)

    def test_gets_not_annotated(self, cui='cui2', name='name-2'):
        fr = FailReason.get_reason_for(cui, name, self.res_w_all, self.tl)
        self.assertIs(fr, FailReason.CONCEPT_NOT_ANNOTATED)
