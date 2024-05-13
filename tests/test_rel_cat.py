import os
import shutil
import unittest
import json

from medcat.cdb import CDB
from medcat.config_rel_cat import ConfigRelCAT
from medcat.rel_cat import RelCAT
from medcat.utils.relation_extraction.rel_dataset import RelData
from medcat.utils.relation_extraction.tokenizer import TokenizerWrapperBERT
from medcat.utils.relation_extraction.models import BertModel_RelationExtraction

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

import spacy
from spacy.tokens import Span, Doc

class RelCATTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config = ConfigRelCAT()
        config.general.device = "cpu"
        config.general.model_name = "bert-base-uncased"
        config.train.batch_size = 1
        config.train.nclasses = 3
        config.model.hidden_size= 256
        config.model.model_size = 2304

        tokenizer = TokenizerWrapperBERT(AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=config.general.model_name,
            config=config), add_special_tokens=True)

        SPEC_TAGS = ["[s1]", "[e1]", "[s2]", "[e2]"]

        tokenizer.hf_tokenizers.add_tokens(SPEC_TAGS, special_tokens=True)
        config.general.annotation_schema_tag_ids = tokenizer.hf_tokenizers.convert_tokens_to_ids(SPEC_TAGS)

        cls.tmp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "tmp")
        os.makedirs(cls.tmp_dir, exist_ok=True)

        cls.save_model_path = os.path.join(cls.tmp_dir, "test_model")
        os.makedirs(cls.save_model_path, exist_ok=True)

        cdb = CDB.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "examples", "cdb.dat"))

        cls.medcat_export_with_rels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "medcat_trainer_export_relations.json")
        cls.medcat_rels_csv_path_train = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "medcat_rel_train.csv")
        cls.medcat_rels_csv_path_test = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", "medcat_rel_test.csv")

        cls.mct_file_test = {}
        with open(cls.medcat_export_with_rels_path, "r+") as f:
            cls.mct_file_test = json.loads(f.read())["projects"][0]["documents"][1]

        cls.config_rel_cat: ConfigRelCAT = config
        cls.rel_cat: RelCAT = RelCAT(cdb, tokenizer=tokenizer, config=config, init_model=True)

        cls.rel_cat.model.bert_model.resize_token_embeddings(len(tokenizer.hf_tokenizers))

        cls.finished = False
        cls.tokenizer = tokenizer

    def test_train_csv_no_tags(self) -> None:
        self.rel_cat.config.train.epochs = 2
        self.rel_cat.train(train_csv_path=self.medcat_rels_csv_path_train, test_csv_path=self.medcat_rels_csv_path_test, checkpoint_path=self.tmp_dir)
        self.rel_cat.save(self.save_model_path)

    def test_train_mctrainer(self) -> None:
        self.rel_cat = RelCAT.load(self.save_model_path)
        self.rel_cat.config.general.mct_export_create_addl_rels = True
        self.rel_cat.config.general.mct_export_max_non_rel_sample_size = 10
        self.rel_cat.config.train.test_size = 0.1
        self.rel_cat.config.train.nclasses = 3
        self.rel_cat.model.relcat_config.train.nclasses = 3
        self.rel_cat.model.bert_model.resize_token_embeddings(len(self.tokenizer.hf_tokenizers))

        self.rel_cat.train(export_data_path=self.medcat_export_with_rels_path, checkpoint_path=self.tmp_dir)

    def test_train_predict(self) -> None:
        Span.set_extension('id', default=0, force=True)
        Span.set_extension('cui', default=None, force=True)
        Doc.set_extension('ents', default=[], force=True)
        Doc.set_extension('relations', default=[], force=True)
        nlp = spacy.blank("en")
        doc = nlp(self.mct_file_test["text"])

        for ann in self.mct_file_test["annotations"]:
            tkn_idx = []
            for ind, word in enumerate(doc):
                end_char = word.idx + len(word.text)
                if end_char <= ann['end'] and end_char > ann['start']:
                    tkn_idx.append(ind)
            entity = Span(doc, min(tkn_idx), max(tkn_idx) + 1, label=ann["value"])
            entity._.cui = ann["cui"]
            doc._.ents.append(entity)

        self.rel_cat.model.bert_model.resize_token_embeddings(len(self.tokenizer.hf_tokenizers))

        doc = self.rel_cat(doc)
        self.finished = True

        assert len(doc._.relations) > 0

    def tearDown(self) -> None:
        if self.finished:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)

if __name__ == '__main__':
    unittest.main()
