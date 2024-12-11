import pydantic
from unittest import TestCase

import pydantic.error_wrappers

from medcat.stats.mctexport import MedCATTrainerExport


MCTExportPydanticModel = pydantic.TypeAdapter(MedCATTrainerExport)



def nullify_doc_names_proj_ids(export: MedCATTrainerExport) -> MedCATTrainerExport:
    return {'projects': [
        {
            'name': project['name'], 
            'documents': sorted([
                {k: v if k != 'name' else '' for k, v in doc.items()} for doc in project['documents']
            ], key=lambda doc: doc['id'])
        } for project in export['projects']
    ]}


def assert_is_mct_export(tc: TestCase, mct_export: dict):
    model_instance = MCTExportPydanticModel.validate_python(mct_export)
    tc.assertIsInstance(model_instance, dict)  # NOTE: otherwise would have raised an exception
