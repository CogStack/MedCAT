import os
import json

from medcat.stats import mctexport

import unittest

from .helpers import MCTExportPydanticModel


class MCTExportIterationTests(unittest.TestCase):
    EXPORT_PATH = os.path.join(os.path.dirname(__file__), "..",
                               "resources", "medcat_trainer_export.json")
    EXPECTED_DOCS = 27
    EXPECTED_ANNS = 435

    @classmethod
    def setUpClass(cls) -> None:
        with open(cls.EXPORT_PATH) as f:
            cls.mct_export = json.load(f)

    def test_conforms_to_template(self):
        # NOTE: This uses pydantic to make sure that the MedCATTrainerExport
        #       type matches the actual export format
        model_instance = MCTExportPydanticModel(**self.mct_export)
        self.assertIsInstance(model_instance, MCTExportPydanticModel)

    def test_iterates_over_all_docs(self):
        self.assertEqual(mctexport.count_all_docs(self.mct_export), self.EXPECTED_DOCS)

    def test_iterates_over_all_anns(self):
        self.assertEqual(mctexport.count_all_annotations(self.mct_export), self.EXPECTED_ANNS)
