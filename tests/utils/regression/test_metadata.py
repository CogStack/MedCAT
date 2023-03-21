from medcat.utils.regression.checking import MetaData

import unittest


# from Anthony's 1.4 model
EXAMPLE_VERSION = 1, 3, 0
MODEL_CARD_EXAMPLE = {
    "Model ID": "acd0dfc2f0df45de",
    "Last Modified On": "04 October 2022",
    "History (from least to most recent)": [
        "3c0f80666505b476",
        "8129693475efdd7e",
        "536f696f7c53eb30",
        "d463dea2cc3d0d9c",
        "17ad6854038dbab7",
        "6d2638c8e9bc6a10",
        "25707e03bd2ffb2d",
        "15ce3df435cafb45"
    ],
    "Description": "This is a UK KCH medcat model. Created on the 20220913. It contains mappings to ICD10 and OPCS4. Enjoy!",
    "Source Ontology": [
        "20220803_SNOMED_UK_clinical_ext",
        "20220831_SNOMED_UK_drug_ext",
        "Enriched via UMLS v2022AA English terms only"
    ],
    "Location": None,
    "MetaCAT models": [],
    "Basic CDB Stats": {
        "Number of concepts": 760334,
        "Number of names": 3081453,
        "Number of concepts that received training": 26143,
        "Number of seen training examples in total": 43578839,
        "Average training examples per concept": 1666.9410167157557
    },
    "Performance": {
        "ner": {},
        "meta": {}
    },
    "Important Parameters (Partial view, all available in cat.config)": {
        "config.ner['min_name_len']": {
            "value": 2,
            "description": "Minimum detection length (found terms/mentions shorter than this will not be detected)."
        },
        "config.ner['upper_case_limit_len']": {
            "value": 2,
            "description": "All detected terms shorter than this value have to be uppercase, otherwise they will be ignored."
        },
        "config.linking['similarity_threshold']": {
            "value": 0.3,
            "description": "If the confidence of the model is lower than this a detection will be ignore."
        },
        "config.linking['filters']['cuis']": {
            "value": 0,
            "description": "Length of the CUIs filter to be included in outputs. If this is not 0 (i.e. not empty) its best to check what is included before using the model"
        },
        "config.general['spell_check']": {
            "value": True,
            "description": "Is spell checking enabled."
        },
        "config.general['spell_check_len_limit']": {
            "value": 6,
            "description": "Words shorter than this will not be spell checked."
        }
    },
    "MedCAT Version": "1.3.0"
}


class MetaDataFromModelCardTest(unittest.TestCase):

    def test_MetaData_creates_from_modelcard(self):
        md = MetaData.from_modelcard(MODEL_CARD_EXAMPLE)
        self.assertIsNotNone(md)

    def test_MetaData_finds_ontology(self):
        md = MetaData.from_modelcard(MODEL_CARD_EXAMPLE)
        self.assertEqual(md.ontology, "SNOMED-CT")

    def test_MetaData_finds_ontology_version(self):
        md = MetaData.from_modelcard(MODEL_CARD_EXAMPLE)
        self.assertEqual(md.ontology_version,
                         "20220803_SNOMED_UK_clinical_ext")
