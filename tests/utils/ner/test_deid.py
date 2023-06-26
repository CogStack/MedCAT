from medcat.utils.ner import deid
from medcat.utils.ner import make_or_update_cdb

from medcat.ner import transformers_ner

from spacy.tokens import Doc

from typing import Any, List, Tuple
import os

import unittest

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


# NB! This 'training data' is extremely flawed
# it is only (somewhat) useful for the purpose of this
# test
# DO NOT USE THIS DATA ELSEWHERE - IT WILL NOT BE USEFUL
TRAIN_DATA = os.path.join(FILE_DIR, "..", "..",
                          "resources", "deid_train_data.json")


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
        self.assertIn("[HOSPITAL]", anon_text)

    def test_model_works_dunder_call(self):
        anon_doc = self.deid_model(input_text)
        self.assertIsInstance(anon_doc, Doc)

    def test_model_works_deid_text_redact(self):
        anon_text = self.deid_model.deid_text(input_text, redact=True)
        self.assertIn("****", anon_text)
        self.assertNotIn("[DOCTOR]", anon_text)
        self.assertNotIn("[HOSPITAL]", anon_text)
