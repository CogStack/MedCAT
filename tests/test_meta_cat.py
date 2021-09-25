import os
import shutil
import unittest

from transformers import AutoTokenizer

from medcat.meta_cat import MetaCAT
from medcat.config_meta_cat import ConfigMetaCAT
from medcat.tokenizers.meta_cat_tokenizers import TokenizerWrapperBERT


class CATTests(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained("bert-base-uncased"))
        config = ConfigMetaCAT()
        config.general['category_name'] = 'Status'
        config.general['cntx_left'] = 15
        config.general['cntx_right'] = 10
        config.general['seed'] = 13
        config.train['nepochs'] = 10
        config.model['input_size'] = 300
        config.train['auto_save_model'] = True

        self.meta_cat = MetaCAT(tokenizer=tokenizer, embeddings=None, config=config)

        self.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)


    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir)


    def test_train(self):
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources', 'mct_export_for_meta_cat_test.json')
        results = self.meta_cat.train(json_path, save_dir_path=self.tmp_dir)

        self.assertEqual(results['f1'], 1.0)


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
