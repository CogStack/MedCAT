import pydantic
from unittest import TestCase

import pydantic.error_wrappers

from medcat.stats.mctexport import MedCATTrainerExport
from medcat.utils.pydantic_version import HAS_PYDANTIC2


if HAS_PYDANTIC2:
    MCTExportPydanticModel = pydantic.TypeAdapter(MedCATTrainerExport)
else:
    MCTExportPydanticModel = pydantic.create_model_from_typeddict(MedCATTrainerExport)



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
    if HAS_PYDANTIC2:
        model_instance = MCTExportPydanticModel.validate_python(mct_export)
        tc.assertIsInstance(model_instance, dict)  # NOTE: otherwise would have raised an exception
    else:
        try:
            model = MCTExportPydanticModel(**mct_export)
        except pydantic.error_wrappers.PydanticValidationError as e:
            raise AssertionError("Not an MCT export") from e
        tc.assertIsInstance(model, MCTExportPydanticModel)
