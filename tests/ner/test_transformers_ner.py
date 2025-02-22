import os
import unittest
from spacy.lang.en import English
from spacy.tokens import Doc, Span
from transformers import TrainerCallback
from medcat.ner.transformers_ner import TransformersNER
from medcat.config import Config
from medcat.cdb_maker import CDBMaker


class TransformerNERTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        config = Config()
        config.general["spacy_model"] = "en_core_web_md"
        cdb_maker = CDBMaker(config)
        cdb_csv = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "examples", "cdb.csv")
        cdb = cdb_maker.prepare_csvs([cdb_csv], full_build=True)
        Doc.set_extension("ents", default=[], force=True)
        Span.set_extension("confidence", default=-1, force=True)
        Span.set_extension("id", default=0, force=True)
        Span.set_extension("detected_name", default=None, force=True)
        Span.set_extension("link_candidates", default=None, force=True)
        Span.set_extension("cui", default=-1, force=True)
        Span.set_extension("context_similarity", default=-1, force=True)
        cls.undertest = TransformersNER(cdb)
        cls.undertest.create_eval_pipeline()

    def test_pipe(self):
        doc = English().make_doc("\nPatient Name: John Smith\nAddress: 15 Maple Avenue\nCity: New York\nCC: Chronic back pain\n\nHX: Mr. Smith")
        doc = next(self.undertest.pipe([doc]))
        assert len(doc.ents) > 0, "No entities were recognised"

    def test_train(self):
        tracker = unittest.mock.Mock()
        class _DummyCallback(TrainerCallback):
            def __init__(self, trainer) -> None:
                self._trainer = trainer
            def on_epoch_end(self, *args, **kwargs) -> None:
                tracker.call()

        train_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "resources", "deid_train_data.json")
        self.undertest.training_arguments.num_train_epochs = 1
        df, examples, dataset = self.undertest.train(train_data, trainer_callbacks=[_DummyCallback, _DummyCallback])
        assert "fp" in examples
        assert "fn" in examples
        assert dataset["train"].num_rows == 48
        assert dataset["test"].num_rows == 12
        self.assertEqual(tracker.call.call_count, 2)

    def test_expand_model_with_concepts(self):
        original_num_labels = self.undertest.model.num_labels
        original_out_features  = self.undertest.model.classifier.out_features
        original_label_map_size = len(self.undertest.tokenizer.label_map)
        cui2preferred_name = {
            "concept_1" : "Preferred Name 1",
            "concept_2" : "Preferred Name 2",
        }

        self.undertest.expand_model_with_concepts(cui2preferred_name)

        assert self.undertest.model.num_labels == original_num_labels + len(cui2preferred_name)
        assert self.undertest.model.classifier.out_features == original_out_features + len(cui2preferred_name)
        assert len(self.undertest.tokenizer.label_map) == original_label_map_size + len(cui2preferred_name)
        assert self.undertest.tokenizer.cui2name.get("concept_1") == "Preferred Name 1"
        assert self.undertest.tokenizer.cui2name.get("concept_2") == "Preferred Name 2"
