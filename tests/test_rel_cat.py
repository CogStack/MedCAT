import os
import shutil
import unittest
import json
import logging

from medcat.cdb import CDB
from medcat.config_rel_cat import ConfigRelCAT
from medcat.rel_cat import RelCAT
from medcat.utils.relation_extraction.bert.tokenizer import BaseTokenizerWrapper_RelationExtraction
from medcat.utils.relation_extraction.rel_dataset import RelData

from transformers.models.auto.tokenization_auto import AutoTokenizer

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
        config.general.log_level = logging.DEBUG

        tokenizer = BaseTokenizerWrapper_RelationExtraction(AutoTokenizer.from_pretrained(
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
        cls.rel_cat: RelCAT = RelCAT(cdb, config=config, init_model=True)

        cls.rel_cat.component.model.hf_model.resize_token_embeddings(len(tokenizer.hf_tokenizers))
        cls.rel_cat.component.model_config.hf_model_config.vocab_size = tokenizer.get_size()

        cls.finished = False
        cls.tokenizer = tokenizer

    def test_dataset_relation_parser(self) -> None:

        samples = [
            "The [s1]45-year-old male[e1] was diagnosed with [s2]hypertension[e2] during his routine check-up.",
            "The patient’s [s1]chest pain[e1] was associated with [s2]shortness of breath[e2].",
            "[s1]Blood pressure[e1] readings of [s2]160/90 mmHg[e2] indicated possible hypertension.",
            "His elevated [s1]blood glucose[e1] level of [s2]220 mg/dL[e2] raised concerns about his diabetes management.",
            "The doctor recommended a [s1]cardiac enzyme test[e1] to assess the risk of [s2]myocardial infarction[e2].",
            "The patient’s [s1]ECG[e1] showed signs of [s2]ischemia[e2]",
            "To manage his [s1]hypertension[e1], the patient was advised to [s2]reduce salt intake[e2].",
            "[s1]Increased physical activity[e1][s2]type 2 diabetes[e2]."
        ]

        rel_dataset = RelData(cdb=self.rel_cat.cdb, config=self.config_rel_cat, tokenizer=self.tokenizer)

        rels = []

        for idx in range(len(samples)):
            tkns = self.tokenizer(samples[idx])["tokens"]
            ent1_ent2_tokens_start_pos = (tkns.index("[s1]"), tkns.index("[s2]"))
            rels.append(rel_dataset.create_base_relations_from_doc(samples[idx], idx,
                                                              ent1_ent2_tokens_start_pos=ent1_ent2_tokens_start_pos))

        self.assertEqual(len(rels), len(samples))

    def test_train_csv_no_tags(self) -> None:
        self.rel_cat.component.relcat_config.train.epochs = 2
        self.rel_cat.train(train_csv_path=self.medcat_rels_csv_path_train, test_csv_path=self.medcat_rels_csv_path_test, checkpoint_path=self.tmp_dir)
        self.rel_cat.save(self.save_model_path)

    def test_train_mctrainer(self) -> None:
        self.rel_cat = RelCAT.load(self.save_model_path)
        self.rel_cat.component.relcat_config.general.create_addl_rels = True
        self.rel_cat.component.relcat_config.general.addl_rels_max_sample_size = 10
        self.rel_cat.component.relcat_config.train.test_size = 0.1
        self.rel_cat.component.relcat_config.train.nclasses = 3

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

        self.rel_cat.component.model.hf_model.resize_token_embeddings(len(self.tokenizer.hf_tokenizers))

        doc = self.rel_cat(doc)
        self.finished = True

        self.assertGreater(len(doc._.relations), 0)


    def tearDown(self) -> None:
        if self.finished:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)


if __name__ == '__main__':
    unittest.main()
