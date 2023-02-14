import os
import shutil
import unittest

from transformers import AutoTokenizer

from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT


class MetaCATTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained('prajjwal1/bert-tiny'))
        config = ConfigMetaCAT()
        config.general['category_name'] = 'Status'
        config.train['nepochs'] = 1
        config.model['input_size'] = 100

        cls.meta_cat = MetaCAT(tokenizer=tokenizer, embeddings=None, config=config)

        cls.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(cls.tmp_dir, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)

    def test_train(self):
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'mct_export_for_meta_cat_test.json')
        results = self.meta_cat.train(json_path, save_dir_path=self.tmp_dir)

        self.assertEqual(results['report']['weighted avg']['f1-score'], 1.0)

    def test_save_load(self):
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'mct_export_for_meta_cat_test.json')
        self.meta_cat.train(json_path, save_dir_path=self.tmp_dir)
        self.meta_cat.save(self.tmp_dir)
        n_meta_cat = MetaCAT.load(self.tmp_dir)

        f1 = self.meta_cat.eval(json_path)['f1']
        n_f1 = n_meta_cat.eval(json_path)['f1']

        self.assertEqual(f1, n_f1)


if __name__ == '__main__':
    unittest.main()
