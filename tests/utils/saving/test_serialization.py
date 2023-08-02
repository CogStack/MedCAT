import os
import glob
import shutil

import tempfile
import unittest

from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.vocab import Vocab

from medcat.utils.saving.serializer import JsonSetSerializer, CDBSerializer, SPECIALITY_NAMES, ONE2MANY

import medcat.utils.saving.coding as _


class JSONSerializationTests(unittest.TestCase):
    folder = os.path.join('temp', 'JSONSerializationTests')

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
            if name in ONE2MANY:
                # ignore cui2many and name2many
                # since they don't exist if/when
                # optimisation hasn't been done
                continue
            with self.subTest(name):
                orig = getattr(self.cdb, name)
                now = getattr(cdb, name)
                self.assertEqual(orig, now)


class ModelCreationTests(unittest.TestCase):
    dill_model_pack = tempfile.TemporaryDirectory()
    json_model_pack = tempfile.TemporaryDirectory()
    EXAMPLES = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "..", "..", "examples")

    @classmethod
    def setUpClass(cls) -> None:
        cls.cdb = CDB.load(os.path.join(cls.EXAMPLES, "cdb.dat"))
        cls.vocab = Vocab.load(os.path.join(cls.EXAMPLES, "vocab.dat"))
        cls.cdb.config.general.spacy_model = "en_core_web_md"
        cls.cdb.config.ner.min_name_len = 2
        cls.cdb.config.ner.upper_case_limit_len = 3
        cls.cdb.config.general.spell_check = True
        cls.cdb.config.linking.train_count_threshold = 10
        cls.cdb.config.linking.similarity_threshold = 0.3
        cls.cdb.config.linking.train = True
        cls.cdb.config.linking.disamb_length_limit = 5
        cls.cdb.config.general.full_unlink = True
        cls.undertest = CAT(
            cdb=cls.cdb, config=cls.cdb.config, vocab=cls.vocab)

    def setUp(self) -> None:
        self.dill_model_pack_name = self.undertest.create_model_pack(
            self.dill_model_pack.name)

    def test_dill_to_json(self):
        model_pack_path = self.undertest.create_model_pack(
            self.json_model_pack.name, cdb_format='json')
        model_pack_folder = os.path.join(
            self.json_model_pack.name, model_pack_path)
        json_path = os.path.join(model_pack_folder, "*.json")
        jsons = glob.glob(json_path)
        # there is also a model_card.json
        # but nothing for cui2many or name2many
        # so can remove the length of ONE2MANY
        self.assertGreaterEqual(len(jsons), len(
            SPECIALITY_NAMES) - len(ONE2MANY))
        for json in jsons:
            with self.subTest(f'JSON {json}'):
                if json.endswith('model_card.json'):
                    continue  # ignore model card here
                if any(name in json for name in ONE2MANY):
                    # ignore cui2many and name2many
                    # since they don't exist if/when
                    # optimisation hasn't been done
                    continue
                self.assertTrue(
                    any(special_name in json for special_name in SPECIALITY_NAMES))
        return model_pack_folder

    def test_load_json(self):
        folder = self.test_dill_to_json()  # make sure the files exist
        cat = CAT.load_model_pack(folder)
        self.assertIsInstance(cat, CAT)

    def test_round_trip(self):
        folder = self.test_dill_to_json()  # make sure the files exist
        cat = CAT.load_model_pack(folder)
        # The spacy model has full path in the loaded model, thus won't be equal
        cat.config.general.spacy_model = os.path.basename(
            cat.config.general.spacy_model)
        # There can also be issues with loading the config.linking.weighted_average_function from file
        # This should be fixed with newer models,
        # but the example model is older, so has the older functionalitys
        cat.config.linking.weighted_average_function = self.undertest.config.linking.weighted_average_function
        self.assertEqual(cat.config.asdict(), self.undertest.config.asdict())
        self.assertEqual(cat.cdb.config, self.undertest.cdb.config)
        self.assertEqual(len(cat.vocab.vocab), len(self.undertest.vocab.vocab))
        for key, vec_and_keys in cat.vocab.vocab.items():
            with self.subTest(f'Vocab.vocab["{key}"]'):
                self.assertIn(key, self.undertest.vocab.vocab)
                other_vec_and_keys = self.undertest.vocab.vocab[key]
                self.assertEqual(len(vec_and_keys), len(other_vec_and_keys))
                for key2, val2 in vec_and_keys.items():
                    with self.subTest(f'Vocab.vocab["{key}"]["{key2}"]'):
                        self.assertIn(key2, other_vec_and_keys)
                        other_val2 = other_vec_and_keys[key2]
                        if hasattr(val2, '__iter__'):
                            self.assertTrue(all(val2 == other_val2))
                        else:
                            self.assertEqual(val2, other_val2)
        self.assertEqual(cat.vocab.index2word, self.undertest.vocab.index2word)
        self.assertEqual(cat.vocab.vec_index2word,
                         self.undertest.vocab.vec_index2word)
        self.assertEqual(cat.vocab.unigram_table,
                         self.undertest.vocab.unigram_table)
        for name in SPECIALITY_NAMES:
            if name in ONE2MANY:
                # ignore cui2many and name2many
                # since they don't exist if/when
                # optimisation hasn't been done
                continue
            with self.subTest(f'CDB Name {name}'):
                self.assertEqual(cat.cdb.__dict__[
                                 name], self.undertest.cdb.__dict__[name])
