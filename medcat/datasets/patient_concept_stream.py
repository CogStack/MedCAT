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
Takes as input a pickled dict of pt2stream. The format should be:
    {'patient_id': (concept_cui, concept_count_for_patient, timestamp_of_first_occurrence_for_patient), ...}
"""


class PatientConceptStreamConfig(datasets.BuilderConfig):
    """BuilderConfig for PatientConceptStream.

        Args:
            **kwargs: keyword arguments forwarded to super.
    """
    pass


class PatientConceptStream(datasets.GeneratorBasedBuilder):
    """PatientConceptStream: as input takes the patient to stream of concepts.

    TODO: Move the preparations scripts out of notebooks
    """

    BUILDER_CONFIGS = [
        PatientConceptStreamConfig(
            name="pickle",
            version=datasets.Version("1.0.0", ""),
            description="Pickled output from Temporal dataset preparation scripts",
        ),
    ]


    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "patient_id": datasets.Value("string"),
                    "stream": [datasets.Value('string')],
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
            pt2stream = pickle.load(f)
            for pt, stream in pt2stream.items():
                out_stream = []
                year = -1
                for data in stream:
                    # 0 - CUI, 1 - CNT, 2 - TIME, 3 - Pt age in Years
                    if data[3] > year:
                        out_stream.append(str(data[3]))
                        year = data[3]

                    out_stream.append(data[0])

                yield pt, {'patient_id': str(pt),
                           'stream': out_stream}
