import unittest
from spacy.lang.en import English
from spacy.tokens import Span
from medcat.pipeline.pipe_runner import PipeRunner


class PipeRunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = "CDB - I was running and then Movar Virus attacked and CDb"
        cls.nlp = English()

    def test_pipe_multi_workers(self):
        docs = list(_PipeRunnerImpl(workers=2).pipe(
            [self.nlp.make_doc(self.text), self.nlp.make_doc(self.text), self.nlp.make_doc(self.text)],
            batch_size=1,
            parallel=True
        ))

        self.assertEqual(3, len(docs))
        self.assertEqual(self.text, docs[0].text)
        self.assertEqual(self.text, docs[1].text)
        self.assertEqual(self.text, docs[2].text)

    def test_pipe_single_worker(self):
        docs = list(_PipeRunnerImpl(workers=1).pipe(
            [self.nlp.make_doc(self.text), self.nlp.make_doc(self.text), self.nlp.make_doc(self.text)],
            batch_size=1,
            parallel=True
        ))

        self.assertEqual(3, len(docs))
        self.assertEqual(self.text, docs[0].text)
        self.assertEqual(self.text, docs[1].text)
        self.assertEqual(self.text, docs[2].text)

    def test_pipe_no_workers(self):
        docs = list(_PipeRunnerImpl(workers=2).pipe(
            [self.nlp.make_doc(self.text), self.nlp.make_doc(self.text), self.nlp.make_doc(self.text)],
            batch_size=1,
            parallel=False
        ))

        self.assertEqual(3, len(docs))
        self.assertEqual(self.text, docs[0].text)
        self.assertEqual(self.text, docs[1].text)
        self.assertEqual(self.text, docs[2].text)

    def test_serialize_entities(self):
        doc = self.nlp.make_doc(self.text)
        doc._.ents = [Span(doc, start=0, end=1, label="test")]
        expected = {
            "confidence": -1,
            "context_similarity": -1,
            "cui": -1,
            "detected_name": None,
            "end": 1,
            "id": 0,
            "label": "test",
            "link_candidates": None,
            "start": 0
        }

        serialized = _PipeRunnerImpl.serialize_entities(doc)

        self.assertEqual(expected, serialized._.ents[0])

    def test_deserialize_entities(self):
        doc = self.nlp.make_doc(self.text)
        doc._.ents = [{
            "confidence": -1,
            "context_similarity": -1,
            "cui": -1,
            "detected_name": None,
            "end": 1,
            "id": 0,
            "label": "test",
            "link_candidates": None,
            "start": 0
        }]

        deserialized = _PipeRunnerImpl.deserialize_entities(doc)._.ents[0]

        self.assertEqual(-1, deserialized._.confidence)
        self.assertEqual(-1, deserialized._.context_similarity)
        self.assertEqual(-1, deserialized._.cui)
        self.assertEqual(None, deserialized._.detected_name)
        self.assertEqual(1, deserialized.end)
        self.assertEqual(0, deserialized._.id)
        self.assertEqual("test", deserialized.label_)
        self.assertEqual(None, deserialized._.link_candidates)
        self.assertEqual(0, deserialized.start)


class _PipeRunnerImpl(PipeRunner):

    def __call__(self, doc):
        return doc

