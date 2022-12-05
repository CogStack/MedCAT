from __future__ import absolute_import, division, print_function

import json
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
Takes as input a json export from medcattrainer."""


class MedCATAnnotationsConfig(datasets.BuilderConfig):
    """BuilderConfig for MedCATNER.

        Args:
          **kwargs: keyword arguments forwarded to super.
    """
    pass


class TransformersDatasetNER(datasets.GeneratorBasedBuilder):
    """MedCATNER: Output of MedCATtrainer"""

    BUILDER_CONFIGS = [
        MedCATAnnotationsConfig(
            name="json",
            version=datasets.Version("1.0.0", ""),
            description="JSON output from MedCATtrainer",
        ),
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "name": datasets.Value("string"),
                    "ent_starts": datasets.Sequence(datasets.Value("int32")),
                    "ent_ends": datasets.Sequence(datasets.Value("int32")),
                    "ent_cuis": datasets.Sequence(datasets.Value("string")),

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
                    "filepaths": self.config.data_files['train'],
                },
            ),
        ]

    def _generate_examples(self, filepaths):
        cnt = 0
        for filepath in filepaths:
            logging.info("generating examples from = %s", filepath)
            with open(filepath, 'r') as f:
                projects = json.load(f)['projects']
                for project in projects:
                    for doc in project['documents']:
                        starts = []
                        ends = []
                        cuis = []
                        for entity in doc['annotations']:
                            if (entity.get('correct', True) or
                               entity.get('manually_created', False) or
                               entity.get('alternative', False)) and not (
                               entity.get('deleted', False) or
                               entity.get('irrelevant', False) or
                               entity.get('killed', False)):

                                starts.append(entity['start'])
                                ends.append(entity['end'])
                                cuis.append(entity['cui'])
                        doc_id = doc.get('id', cnt)
                        cnt += 1
                        doc_name = doc.get('name', 'unknown')
                        yield "{}".format(doc_id), {
                                'id': int(doc_id),
                                'text': str(doc['text']),
                                'name': str(doc_name),
                                'ent_starts': starts,
                                'ent_ends': ends,
                                'ent_cuis': cuis,
                                }
