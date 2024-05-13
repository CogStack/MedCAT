from typing import Optional, Set, Dict

from medcat.config import LinkingFilters
from medcat.utils.matutils import intersect_nonempty_set


def check_filters(cui: str, filters: LinkingFilters):
    """Checks is a CUI in the filters

    Args:
        cui (str): The CUI.
        filters (LinkingFilters): The filters.

    Returns:
        bool:
            True if in filters else False
    """
    if cui in filters.cuis or not filters.cuis:
        return cui not in filters.cuis_exclude
    else:
        return False


def get_all_irrelevant_cuis(project):
    i_cuis = set()
    for d in project['documents']:
        for a in d['annotations']:
            if 'irrelevant' in a and a['irrelevant']:
                i_cuis.add(a['cui'])
    return i_cuis


def get_project_filters(cuis, type_ids, addl_info: Dict, project=None):
    cui_filter = set()
    if isinstance(cuis, str):
        if cuis is not None and cuis:
            cui_filter.update([x.strip() for x in cuis.split(",")])
        if type_ids is not None and type_ids:
            type_ids = [x.strip().upper() for x in type_ids.split(",")]

            # Convert type_ids to cuis
            if 'type_id2cuis' in addl_info:
                for type_id in type_ids:
                    if type_id in addl_info['type_id2cuis']:
                        cui_filter.update(addl_info['type_id2cuis'][type_id])
                    else:
                        raise Exception("Impossible to create filters, disable them.")
            else:
                raise Exception("Impossible to create filters, disable them.")
    elif isinstance(cuis, list):
        cui_filter = set(cuis)

    if project is not None:
        i_cuis = get_all_irrelevant_cuis(project)
        for i_cui in i_cuis:
            cui_filter.remove(i_cui)

    return cui_filter


def set_project_filters(addl_info: Dict, local_filters: LinkingFilters, project: dict,
            extra_cui_filter: Optional[Set], use_project_filters: bool):
    """Set the project filters to a LinkingFilters object based on
    the specified project.

    Args:
        addl_info (Dict): The CDB additional information
        local_filters (LinkingFilters): The linking filters instance
        project (dict): The project
        extra_cui_filter (Optional[Set]): Extra CUIs (if specified)
        use_project_filters (bool): Whether to use per-project filters
    """
    if isinstance(extra_cui_filter, set):
        local_filters.cuis = extra_cui_filter

    if use_project_filters:
        project_filter = get_project_filters(cuis=project.get('cuis', None),
                                             type_ids=project.get('tuis', None),
                                             addl_info=addl_info,
                                             project=project)
        # Intersect project filter with existing if it has something
        if project_filter:
            local_filters.cuis = intersect_nonempty_set(project_filter, local_filters.cuis)
