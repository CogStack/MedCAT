from typing import List, Iterator, Tuple, Any, Optional
from typing_extensions import TypedDict


class MedCATTrainerExportAnnotation(TypedDict):
    start: int
    end: int
    cui: str
    value: str


class MedCATTrainerExportDocument(TypedDict):
    name: str
    id: Any
    last_modified: str
    text: str
    annotations: List[MedCATTrainerExportAnnotation]


class MedCATTrainerExportProject(TypedDict):
    name: str
    id: Any
    cuis: str
    tuis: Optional[str]
    documents: List[MedCATTrainerExportDocument]


MedCATTrainerExportProjectInfo = Tuple[str, Any, str, Optional[str]]
"""The project name, project ID, CUIs str, and TUIs str"""


class MedCATTrainerExport(TypedDict):
    projects: List[MedCATTrainerExportProject]


def iter_projects(export: MedCATTrainerExport) -> Iterator[MedCATTrainerExportProject]:
    yield from export['projects']


def iter_docs(export: MedCATTrainerExport
              ) -> Iterator[Tuple[MedCATTrainerExportProjectInfo, MedCATTrainerExportDocument]]:
    for project in iter_projects(export):
        info: MedCATTrainerExportProjectInfo = (
            project['name'], project['id'], project['cuis'], project.get('tuis', None)
        )
        for doc in project['documents']:
            yield info, doc


def iter_anns(export: MedCATTrainerExport
              ) -> Iterator[Tuple[MedCATTrainerExportProjectInfo, MedCATTrainerExportDocument, MedCATTrainerExportAnnotation]]:
    for proj_info, doc in iter_docs(export):
        for ann in doc['annotations']:
            yield proj_info, doc, ann


def count_all_annotations(export: MedCATTrainerExport) -> int:
    return len(list(iter_anns(export)))


def count_all_docs(export: MedCATTrainerExport) -> int:
    return len(list(iter_docs(export)))


def get_nr_of_annotations(doc: MedCATTrainerExportDocument) -> int:
    return len(doc['annotations'])
