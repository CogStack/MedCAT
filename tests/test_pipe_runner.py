import unittest
from spacy.lang.en import English
from medcat.pipeline.pipe_runner import PipeRunner


class PipeRunnerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = "CDB - I was running and then Movar Virus attacked and CDb"
        cls.nlp = English()

    def test_pipe_single_process_multi_workers(self):
        docs = list(_PipeRunnerImpl(workers=2, batch_size=1).pipe(
            [self.nlp.make_doc(self.text), self.nlp.make_doc(self.text), self.nlp.make_doc(self.text)],
            parallel=True
        ))

        self.assertEqual(3, len(docs))
        self.assertEqual(self.text, docs[0].text)
        self.assertEqual(self.text, docs[1].text)
        self.assertEqual(self.text, docs[2].text)

    def test_pipe_multi_processes_single_work(self):
        docs = list(_PipeRunnerImpl(workers=1, batch_size=1).pipe(
            [self.nlp.make_doc(self.text), self.nlp.make_doc(self.text), self.nlp.make_doc(self.text)],
            parallel=False
        ))

        self.assertEqual(3, len(docs))
        self.assertEqual(self.text, docs[0].text)
        self.assertEqual(self.text, docs[1].text)
        self.assertEqual(self.text, docs[2].text)

    def test_pipe_multi_processes_multi_workers(self):
        docs = list(_PipeRunnerImpl(workers=2, batch_size=1).pipe(
            [self.nlp.make_doc(self.text), self.nlp.make_doc(self.text), self.nlp.make_doc(self.text)],
            parallel=True
        ))

        self.assertEqual(3, len(docs))
        self.assertEqual(self.text, docs[0].text)
        self.assertEqual(self.text, docs[1].text)
        self.assertEqual(self.text, docs[2].text)


class _PipeRunnerImpl(PipeRunner):

    def __call__(self, doc):
        return doc

