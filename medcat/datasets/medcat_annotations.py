from __future__ import absolute_import, division, print_function

import pickle
import logging

import datasets


_CITATION = """\
@misc{kraljevic2020multidomain,
      title={Multi-domain Clinical Natural Language Processing with MedCAT: the Medical Concept Annotation Toolkit}, 
      author={Zeljko Kraljevic and Thomas Searle and Anthony Shek and Lukasz Roguski and Kawsar Noor and Daniel Bean and Aurelie Mascio and Leilei Zhu and Amos A Folarin and Angus Roberts and Rebecca Bendayan and Mark P Richardson and Robert Stewart and Anoop D Shah and Wai Keong Wong and Zina Ibrahim and James T Teo and Richard JB Dobson},
      year={2020},
      eprint={2010.01165},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
Takes as input a pickled dict of annotated documents from MedCAT. The format should be:
    {'document_id': {'entities': <entities>, ...}
Where entities is the output from medcat.get_entities(<...>)['entities']
"""

class MedCATAnnotationsConfig(datasets.BuilderConfig):
    """BuilderConfig for MedCATAnnotations."""

    def __init__(self, **kwargs):
        """BuilderConfig for MedCATAnnotations.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MedCATAnnotationsConfig, self).__init__(**kwargs)


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

    def _split_generators(self, dl_manager):
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
                for id, entity in doc['entities'].items():
                    yield "{}|{}".format(doc_id, entity['id']), {
                            'id': int(id),
                            'document_id': str(doc_id),
                            'context_left': "".join(entity['context_left']),
                            'context_right': "".join(entity['context_right']),
                            'context_center': "".join(entity['context_center']),
                            }
