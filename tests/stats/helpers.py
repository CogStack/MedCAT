from pydantic import create_model_from_typeddict

from medcat.stats.mctexport import MedCATTrainerExport


MCTExportPydanticModel = create_model_from_typeddict(MedCATTrainerExport)


def nullify_doc_names_proj_ids(export: MedCATTrainerExport) -> MedCATTrainerExport:
    return {'projects': [
        {
            'name': project['name'], 
            'documents': sorted([
                {k: v if k != 'name' else '' for k, v in doc.items()} for doc in project['documents']
            ], key=lambda doc: doc['id'])
        } for project in export['projects']
    ]}
