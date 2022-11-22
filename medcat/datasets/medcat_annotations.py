from __future__ import absolute_import, division, print_function

import pickle
import logging

import datasets


_CITATION = """\
@ARTICLE{Kraljevic2021-ln,
  title="Multi-domain clinical natural language processing with {MedCAT}: The Medical Concept Annotation Toolkit",
  author="Kraljevic, Zeljko and Searle, Thomas and Shek, Anthony and Roguski, Lukasz and Noor, Kawsar and Bean, Daniel and Mascio, Aurelie and Zhu, Leilei and Folarin, Amos A and Roberts, Angus and Bendayan, Rebecca and Richardson, Mark P and Stewart, Robert and Shah, Anoop D and Wong, Wai Keong and Ibrahim, Zina and Teo, James T and Dobson, Richard J B",
  journal="Artif. Intell. Med.",
  volume=117,
  pages="102083",
  month=jul,
  year=2021,
  issn="0933-3657",
  doi="10.1016/j.artmed.2021.102083"
}
"""

_DESCRIPTION = """\
Takes as input a pickled dict of annotated documents from MedCAT. The format should be:
    {'document_id': {'entities': <entities>, ...}
Where entities is the output from medcat.get_entities(<...>)['entities']
"""


class MedCATAnnotationsConfig(datasets.BuilderConfig):
    """BuilderConfig for MedCATAnnotations.

        Args:
          **kwargs: keyword arguments forwarded to super.
    """
    pass


class MedCATAnnotations(datasets.GeneratorBasedBuilder):
    """MedCATAnnotations: Output of MedCAT"""

    BUILDER_CONFIGS = [
        MedCATAnnotationsConfig(
            name="pickle",
            version=datasets.Version("1.0.0", ""),
            description="Pickled output from MedCAT",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "document_id": datasets.Value("string"),
                    "context_left": datasets.Value("string"),
                    "context_right": datasets.Value("string"),
                    "context_center": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager): # noqa
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": self.config.data_files,
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        with open(filepath, 'rb') as f:
            docs = pickle.load(f)
            for doc_id in docs:
                doc = docs[doc_id]
                for entity_id, entity in doc['entities'].items():
                    yield "{}|{}".format(doc_id, entity['id']), {
                            'id': int(entity_id),
                            'document_id': str(doc_id),
                            'context_left': "".join(entity['context_left']),
                            'context_right': "".join(entity['context_right']),
                            'context_center': "".join(entity['context_center']),
                            }
