from medcat.utils.ner import deid
from medcat.utils.ner import make_or_update_cdb

from medcat.ner import transformers_ner

from spacy.tokens import Doc, Span

from typing import Any, List, Tuple
import os
import json
import tempfile

import unittest
import timeout_decorator

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


# NB! This 'training data' is extremely flawed
# it is only (somewhat) useful for the purpose of this
# test
# DO NOT USE THIS DATA ELSEWHERE - IT WILL NOT BE USEFUL
TRAIN_DATA = os.path.join(FILE_DIR, "..", "..",
                          "resources", "deid_train_data.json")

TEST_DATA = os.path.join(FILE_DIR, "..", "..",
                         "resources", "deid_test_data.json")


class DeIDmodelCreationTests(unittest.TestCase):

    def test_can_make_cdb(self):
        cdb = make_or_update_cdb(TRAIN_DATA)
        self.assertIsNotNone(cdb)

    def test_can_create_model(self):
        cdb = make_or_update_cdb(TRAIN_DATA)
        config = transformers_ner.ConfigTransformersNER()
        config.general['test_size'] = 0.1  # Usually set this to 0.1-0.2
        ner = transformers_ner.TransformersNER(cdb=cdb, config=config)
        deid_model = deid.DeIdModel.create(ner)
        self.assertIsNotNone(deid_model)


def _add_model(cls):
    cdb = make_or_update_cdb(TRAIN_DATA)
    config = transformers_ner.ConfigTransformersNER()
    config.general['test_size'] = 0.1  # Usually set this to 0.1-0.2
    config.general['chunking_overlap_window'] = None
    cls.ner = transformers_ner.TransformersNER(cdb=cdb, config=config)
    cls.ner.training_arguments.num_train_epochs = 1  # Use 5-10 normally
    # As we are NOT training on a GPU that can, we'll set it to 1
    cls.ner.training_arguments.per_device_train_batch_size = 1
    cls.ner.training_arguments.gradient_accumulation_steps = 1  # No need for acc
    cls.ner.training_arguments.per_device_eval_batch_size = 1
    # For the metric to be used for best model we pick Recall here, as for deid that is most important
    cls.ner.training_arguments.metric_for_best_model = 'eval_recall'
    cls.deid_model = deid.DeIdModel.create(cls.ner)


def train_model_once(model: deid.DeIdModel,
                     _trained: List[Tuple[Tuple[Any, Any, Any],
                                          deid.DeIdModel]] = []
                     ) -> Tuple[Tuple[Any, Any, Any], deid.DeIdModel]:
    if not _trained:
        retval = model.train(TRAIN_DATA)
        # mpp = 'temp/deid_multiprocess/dumps/temp_model_save'
        # NOTE: it seems that after training the model leaves
        #       it in a state where it can no longer be used
        #       for multiprocessing. So in order to avoid that
        #       we save the model on disk and load it agains
        with tempfile.TemporaryDirectory() as dir_name:
            print("Saving model on disk")
            mpn = model.cat.create_model_pack(dir_name)
            print("Loading model")
            model = deid.DeIdModel.load_model_pack(os.path.join(dir_name, mpn))
            print("Loaded model off disk")
        _trained.append((retval, model))
    return _trained[0]


class DeIDModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _add_model(cls)

    def test_training(self):
        df, examples, dataset = train_model_once(self.deid_model)[0]
        self.assertIsNotNone(df)
        self.assertIsNotNone(examples)
        self.assertIsNotNone(dataset)

    def test_add_new_concepts(self):
        self.deid_model.add_new_concepts({'CONCEPT': "Concept"}, with_random_init=True)
        self.assertTrue("CONCEPT" in self.deid_model.cat.cdb.cui2names)
        self.assertEqual(self.deid_model.cat.cdb.cui2names["CONCEPT"], {"concept"})
        self.assertTrue("CONCEPT" in self.deid_model.cat._addl_ner[0].model.config.label2id)
        self.assertTrue("CONCEPT" in self.deid_model.cat._addl_ner[0].tokenizer.label_map)
        self.assertTrue("CONCEPT" in self.deid_model.cat._addl_ner[0].tokenizer.cui2name)


input_text = '''
James Joyce
7 Eccles Street,
Dublin
CC: Memory difficulty.

HX: Mr James is a 64 y/o RHM, had difficulty remembering names, phone numbers and events for 12 months prior to presentation, on 2/28/95. He had visited London recently and had had no professional or social faux pas or mishaps due to his memory. J.J. could not tell whether his problem was becoming worse, so he brought himself to the Neurology clinic on his own referral.

FHX: Both parents (Mary and John) experienced memory problems in their ninth decades, but not earlier. 5 siblings have had no memory trouble. There are no neurological illnesses in his family.

SHX: Writer and Poet. Tobacco/ETOH/illicit drug use.

The rest of the neurologic exam was unremarkable and there were no extrapyramidal signs or primitive reflexes noted.
11/1996 in Dublin.

The findings indicated multiple areas of cerebral dysfunction. With the exception of the patient's report of minimal occupational dysfunction ( which may reflect poor insight), the clinical picture is consistent with a progressive dementia syndrome such as Alzheimer disease. MRI brain, 3/6/95, showed mild generalized atrophy, more severe in the occipital-parietal regions.

Seen by Dr. M. Sully on 11/11/1996.
'''


class DeIDModelWorks(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _add_model(cls)
        cls.deid_model = train_model_once(cls.deid_model)[1]

    def test_model_works_deid_text(self):
        anon_text = self.deid_model.deid_text(input_text)
        self.assertIn("[DOCTOR]", anon_text)
        self.assertNotIn("M. Sully", anon_text)
        self.assertIn("[HOSPITAL]", anon_text)
        # self.assertNotIn("Dublin", anon_text)
        self.assertNotIn("7 Eccles Street", anon_text)

    def test_model_works_dunder_call(self):
        anon_doc = self.deid_model(input_text)
        self.assertIsInstance(anon_doc, Doc)

    def test_model_works_deid_text_redact(self):
        anon_text = self.deid_model.deid_text(input_text, redact=True)
        self.assertIn("****", anon_text)
        self.assertNotIn("[DOCTOR]", anon_text)
        self.assertNotIn("M. Sully", anon_text)
        self.assertNotIn("[HOSPITAL]", anon_text)
        # self.assertNotIn("Dublin", anon_text)
        self.assertNotIn("7 Eccles Street", anon_text)


class DeIDModelMultiprocessingWorks(unittest.TestCase):
    processes = 2

    @classmethod
    def setUpClass(cls) -> None:
        Span.set_extension('link_candidates', default=None, force=True)
        _add_model(cls)
        cls.deid_model = train_model_once(cls.deid_model)[1]
        with open(TEST_DATA) as f:
            raw_data = json.load(f)
        cls.data = []
        for project in raw_data['projects']:
            for doc in project['documents']:
                cls.data.append((f"{project['name']}_{doc['name']}", doc['text']))
        # NOTE: Comment and subsequent code
        #       copied from CAT.multiprocessing_batch_char_size
        #       (lines 1234 - 1237)
        # Hack for torch using multithreading, which is not good if not
        #separate_nn_components, need for CPU runs only
        import torch
        torch.set_num_threads(1)

    def assertTextHasBeenDeIded(self, text: str, redacted: bool):
        if not redacted:
            for cui in self.deid_model.cdb.cui2names:
                cui_name = self.deid_model.cdb.get_name(cui)
                if cui_name in text:
                    # all good
                    return
        else:
            # if redacted, only check once...
            if "******" in text:
                # all good
                return
        raise AssertionError("None of the CUIs found")

    @timeout_decorator.timeout(3 * 60)  # 3 minutes max
    def test_model_can_multiprocess_no_redact(self):
        processed = self.deid_model.deid_multi_texts(self.data, n_process=self.processes)
        self.assertEqual(len(processed), 5)
        for tid, new_text in enumerate(processed):
            with self.subTest(str(tid)):
                self.assertTextHasBeenDeIded(new_text, redacted=False)

    @timeout_decorator.timeout(3 * 60)  # 3 minutes max
    def test_model_can_multiprocess_redact(self):
        processed = self.deid_model.deid_multi_texts(self.data, n_process=self.processes, redact=True)
        self.assertEqual(len(processed), 5)
        for tid, new_text in enumerate(processed):
            with self.subTest(str(tid)):
                self.assertTextHasBeenDeIded(new_text, redacted=True)


class MatchRulesTests(unittest.TestCase):
    def test_match_rules(self):
        # Test data from the docstring example
        rules = [
            ('(123) 456-7890', '134'),
            ('1234567890', '134'),
            ('123.456.7890', '134'),
            ('1234567890', '134'),
            ('1234567890', '134'),
        ]
        texts = [
            'My phone number is (123) 456-7890',
            'My phone number is 1234567890',
            'My phone number is 123.456.7890',
            'My phone number is 1234567890',
        ]
        cui2preferred_name = {'134': 'Phone Number'}

        # Get matches
        matches = deid.match_rules(rules, texts, cui2preferred_name)

        # Verify results
        self.assertEqual(len(matches), len(texts))  # One list of matches per text

        # Check first text matches
        self.assertEqual(len(matches[0]), 1)  # One match in first text
        self.assertEqual(matches[0][0]['source_value'], '(123) 456-7890')
        self.assertEqual(matches[0][0]['pretty_name'], 'Phone Number')
        self.assertEqual(matches[0][0]['cui'], '134')
        self.assertEqual(matches[0][0]['acc'], 1.0)
        self.assertEqual(matches[0][0]['start'], 17)  # Position of phone number in text
        self.assertEqual(matches[0][0]['end'], 31)  # End position of phone number

        # Check second text matches
        self.assertEqual(len(matches[1]), 1)  # One match in second text
        self.assertEqual(matches[1][0]['source_value'], '1234567890')
        self.assertEqual(matches[1][0]['pretty_name'], 'Phone Number')
        self.assertEqual(matches[1][0]['cui'], '134')
        self.assertEqual(matches[1][0]['acc'], 1.0)
        self.assertEqual(matches[1][0]['start'], 17)  # Position of phone number in text
        self.assertEqual(matches[1][0]['end'], 27)  # End position of phone number

        # Check third text matches
        self.assertEqual(len(matches[2]), 1)  # One match in third text
        self.assertEqual(matches[2][0]['source_value'], '123.456.7890')
        self.assertEqual(matches[2][0]['pretty_name'], 'Phone Number')
        self.assertEqual(matches[2][0]['cui'], '134')
        self.assertEqual(matches[2][0]['acc'], 1.0)
        self.assertEqual(matches[2][0]['start'], 17)  # Position of phone number in text
        self.assertEqual(matches[2][0]['end'], 30)  # End position of phone number

        # Check fourth text matches
        self.assertEqual(len(matches[3]), 1)  # One match in fourth text
        self.assertEqual(matches[3][0]['source_value'], '1234567890')
        self.assertEqual(matches[3][0]['pretty_name'], 'Phone Number')
        self.assertEqual(matches[3][0]['cui'], '134')
        self.assertEqual(matches[3][0]['acc'], 1.0)
        self.assertEqual(matches[3][0]['start'], 17)  # Position of phone number in text
        self.assertEqual(matches[3][0]['end'], 27)  # End position of phone number

    def test_merge_preds(self):
        # Test data with overlapping predictions
        model_preds = [
            {'cui': '134', 'start': 10, 'end': 20, 'acc': 0.9,
             'pretty_name': 'Phone Number'},
            {'cui': '134', 'start': 25, 'end': 35, 'acc': 0.8,
             'pretty_name': 'Phone Number'},
            {'cui': '134', 'start': 50, 'end': 60, 'acc': 0.9,  # Non-overlapping model pred
             'pretty_name': 'Phone Number'}
        ]
        rule_matches = [
            {'cui': '134', 'start': 15, 'end': 25, 'acc': 1.0,  # Overlaps with first model pred
             'pretty_name': 'Phone Number'},
            {'cui': '134', 'start': 30, 'end': 40, 'acc': 1.0,  # Overlaps with second model pred
             'pretty_name': 'Phone Number'},
            {'cui': '134', 'start': 70, 'end': 80, 'acc': 1.0,  # Non-overlapping rule match
             'pretty_name': 'Phone Number'}
        ]

        # Test with accept_preds=True (default)
        merged_preds = deid.merge_preds(model_preds, rule_matches)
        self.assertEqual(len(merged_preds), 1)  # Should return a list with one element
        self.assertEqual(len(merged_preds[0]), 3)  # Should keep model predictions and non-overlapping rule match
        self.assertEqual(merged_preds[0][0]['start'], 10)  # First model pred
        self.assertEqual(merged_preds[0][1]['start'], 25)  # Second model pred
        self.assertEqual(merged_preds[0][2]['start'], 50)  # Third model pred

        # Test with accept_preds=False
        merged_preds = deid.merge_preds(model_preds, rule_matches, accept_preds=False)
        self.assertEqual(len(merged_preds), 1)  # Should return a list with one element
        self.assertEqual(len(merged_preds[0]), 3)  # Should keep rule matches and non-overlapping model pred
        self.assertEqual(merged_preds[0][0]['start'], 15)  # First rule match
        self.assertEqual(merged_preds[0][1]['start'], 30)  # Second rule match
        self.assertEqual(merged_preds[0][2]['start'], 70)  # Third rule match

        # Test with non-overlapping predictions
        model_preds = [
            {'cui': '134', 'start': 10, 'end': 20, 'acc': 0.9,
             'pretty_name': 'Phone Number'},
            {'cui': '134', 'start': 50, 'end': 60, 'acc': 0.9,  # Additional non-overlapping model pred
             'pretty_name': 'Phone Number'}
        ]
        rule_matches = [
            {'cui': '134', 'start': 25, 'end': 35, 'acc': 1.0,
             'pretty_name': 'Phone Number'},
            {'cui': '134', 'start': 70, 'end': 80, 'acc': 1.0,  # Additional non-overlapping rule match
             'pretty_name': 'Phone Number'}
        ]

        # Test with accept_preds=True (default)
        merged_preds = deid.merge_preds(model_preds, rule_matches)
        self.assertEqual(len(merged_preds), 1)
        self.assertEqual(len(merged_preds[0]), 4)  # Should keep all predictions
        self.assertEqual(merged_preds[0][0]['start'], 10)  # First model pred
        self.assertEqual(merged_preds[0][1]['start'], 25)  # First rule match
        self.assertEqual(merged_preds[0][2]['start'], 50)  # Second model pred
        self.assertEqual(merged_preds[0][3]['start'], 70)  # Second rule match

        # Test with accept_preds=False
        merged_preds = deid.merge_preds(model_preds, rule_matches, accept_preds=False)
        self.assertEqual(len(merged_preds), 1)
        self.assertEqual(len(merged_preds[0]), 4)  # Should keep all predictions
        self.assertEqual(merged_preds[0][0]['start'], 10)  # First model pred
        self.assertEqual(merged_preds[0][1]['start'], 25)  # First rule match
        self.assertEqual(merged_preds[0][2]['start'], 50)  # Second model pred
        self.assertEqual(merged_preds[0][3]['start'], 70)  # Second rule match

    def test_merge_all_preds(self):
        # Test with lists of different lengths
        model_preds_by_text = [
            [{'cui': '134', 'start': 10, 'end': 20, 'acc': 0.9, 'pretty_name': 'Phone Number'}],
            [{'cui': '134', 'start': 25, 'end': 35, 'acc': 0.8, 'pretty_name': 'Phone Number'}]
        ]
        rule_matches_per_text = [
            [{'cui': '134', 'start': 15, 'end': 25, 'acc': 1.0, 'pretty_name': 'Phone Number'}]
        ]

        # Test that it raises ValueError for different lengths
        with self.assertRaises(AssertionError) as context:
            deid.merge_all_preds(model_preds_by_text, rule_matches_per_text)
        self.assertIn("must have the same length", str(context.exception))

        # Test with consistent lengths
        model_preds_by_text = [
            [{'cui': '134', 'start': 10, 'end': 20, 'acc': 0.9, 'pretty_name': 'Phone Number'}],
            [{'cui': '134', 'start': 25, 'end': 35, 'acc': 0.8, 'pretty_name': 'Phone Number'}]
        ]
        rule_matches_per_text = [
            [{'cui': '134', 'start': 15, 'end': 25, 'acc': 1.0, 'pretty_name': 'Phone Number'}],
            [{'cui': '134', 'start': 30, 'end': 40, 'acc': 1.0, 'pretty_name': 'Phone Number'}]
        ]

        # Test with accept_preds=True (default)
        merged_preds = deid.merge_all_preds(model_preds_by_text, rule_matches_per_text)
        self.assertEqual(len(merged_preds), 2)  # Two texts
        self.assertEqual(len(merged_preds[0]), 1)  # First text has one model pred
        self.assertEqual(len(merged_preds[1]), 1)  # Second text has one model pred
        self.assertEqual(merged_preds[0][0]['start'], 10)  # First text model pred
        self.assertEqual(merged_preds[1][0]['start'], 25)  # Second text model pred

        # Test with accept_preds=False
        merged_preds = deid.merge_all_preds(model_preds_by_text, rule_matches_per_text, accept_preds=False)
        self.assertEqual(len(merged_preds), 2)  # Two texts
        self.assertEqual(len(merged_preds[0]), 1)  # First text has one rule match
        self.assertEqual(len(merged_preds[1]), 1)  # Second text has one rule match
        self.assertEqual(merged_preds[0][0]['start'], 15)  # First text rule match
        self.assertEqual(merged_preds[1][0]['start'], 30)  # Second text rule match

        # Test with non-overlapping predictions
        model_preds_by_text = [
            [{'cui': '134', 'start': 10, 'end': 20, 'acc': 0.9, 'pretty_name': 'Phone Number'}],
            [{'cui': '134', 'start': 25, 'end': 35, 'acc': 0.8, 'pretty_name': 'Phone Number'}]
        ]
        rule_matches_per_text = [
            [{'cui': '134', 'start': 30, 'end': 40, 'acc': 1.0, 'pretty_name': 'Phone Number'}],
            [{'cui': '134', 'start': 50, 'end': 60, 'acc': 1.0, 'pretty_name': 'Phone Number'}]
        ]

        # Test with accept_preds=True (default)
        merged_preds = deid.merge_all_preds(model_preds_by_text, rule_matches_per_text)
        self.assertEqual(len(merged_preds), 2)  # Two texts
        self.assertEqual(len(merged_preds[0]), 2)  # First text has both preds
        self.assertEqual(len(merged_preds[1]), 2)  # Second text has both preds
        self.assertEqual(merged_preds[0][0]['start'], 10)  # First text model pred
        self.assertEqual(merged_preds[0][1]['start'], 30)  # First text rule match
        self.assertEqual(merged_preds[1][0]['start'], 25)  # Second text model pred
        self.assertEqual(merged_preds[1][1]['start'], 50)  # Second text rule match
