import os
import shutil
import unittest

from transformers import AutoTokenizer

from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT
import spacy
from spacy.tokens import Span


class MetaCATTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'))
        config = ConfigMetaCAT()
        config.general['category_name'] = 'Status'
        config.train['nepochs'] = 2
        config.model['input_size'] = 100

        cls.meta_cat: MetaCAT = MetaCAT(tokenizer=tokenizer, embeddings=None, config=config)

        cls.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(cls.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_train(self):
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources',
                                 'mct_export_for_meta_cat_test.json')
        results = self.meta_cat.train_from_json(json_path, save_dir_path=self.tmp_dir)
        if self.meta_cat.config.model.phase_number != 1:
            self.assertEqual(results['report']['weighted avg']['f1-score'], 1.0)

    def test_save_load(self):
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources',
                                 'mct_export_for_meta_cat_test.json')
        self.meta_cat.train_from_json(json_path, save_dir_path=self.tmp_dir)
        self.meta_cat.save(self.tmp_dir)
        n_meta_cat = MetaCAT.load(self.tmp_dir)

        f1 = self.meta_cat.eval(json_path)['f1']
        n_f1 = n_meta_cat.eval(json_path)['f1']

        self.assertEqual(f1, n_f1)

    def _prepare_doc_w_spangroup(self, spangroup_name: str):
        """
        Create spans under an arbitrary spangroup key
        """
        Span.set_extension('id', default=0, force=True)
        Span.set_extension('meta_anns', default=None, force=True)
        nlp = spacy.blank("en")
        doc = nlp("Pt has diabetes and copd.")
        span_0 = doc.char_span(7, 15, label="diabetes")
        assert span_0.text == 'diabetes'

        span_1 = doc.char_span(20, 24, label="copd")
        assert span_1.text == 'copd'
        doc.spans[spangroup_name] = [span_0, span_1]
        return doc

    def test_predict_spangroup(self):
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources',
                                 'mct_export_for_meta_cat_test.json')
        self.meta_cat.train_from_json(json_path, save_dir_path=self.tmp_dir)
        self.meta_cat.save(self.tmp_dir)
        n_meta_cat = MetaCAT.load(self.tmp_dir)

        spangroup_name = "mock_span_group"
        n_meta_cat.config.general.span_group = spangroup_name

        doc = self._prepare_doc_w_spangroup(spangroup_name)
        doc = n_meta_cat(doc)
        spans = doc.spans[spangroup_name]
        self.assertEqual(len(spans), 2)

        # All spans are annotate
        for span in spans:
            self.assertEqual(span._.meta_anns['Status']['value'], "Affirmed")

        # Informative error if spangroup is not set
        doc = self._prepare_doc_w_spangroup("foo")
        n_meta_cat.config.general.span_group = "bar"
        try:
            doc = n_meta_cat(doc)
        except Exception as error:
            self.assertIn("Configuration error", str(error))

        n_meta_cat.config.general.span_group = None


class MetaCATBertTest(MetaCATTests):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'))
        config = ConfigMetaCAT()
        config.general['category_name'] = 'Status'
        config.train['nepochs'] = 2
        config.model['input_size'] = 100
        config.train['batch_size'] = 64
        config.model['model_name'] = 'bert'

        cls.meta_cat: MetaCAT = MetaCAT(tokenizer=tokenizer, embeddings=None, config=config)
        cls.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(cls.tmp_dir, exist_ok=True)

    def test_two_phase(self):
        self.meta_cat.config.model['phase_number'] = 1
        self.test_train()
        self.meta_cat.config.model['phase_number'] = 2
        self.test_train()

        self.meta_cat.config.model['phase_number'] = 0


if __name__ == '__main__':
    unittest.main()
