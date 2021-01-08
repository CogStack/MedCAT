def check_filters(cui, filters):
    r''' Checks is a CUI in the filters

    Args:
        cui
        filters

    Return:
        bool:
            True if in filters else False
    '''
    # There is only one filter for now (inclusion)
    if not filters['cuis']:
        # If filter is empty use everything
        return True
    elif cui in filters['cuis']:
        # CUI is in filters
        return True
    else:
        return False

def process_old_project_filters(cuis, type_ids, cdb):
    cui_filter = set()
    if cuis is not None and cuis:
        cui_filter.update([x.strip() for x in cuis.split(",")])
    if type_ids is not None and type_ids:
        type_ids = [x.strip().upper() for x in type_ids.split(",")]

        # Convert type_ids to cuis
        if 'type_id2cuis' in cdb.addl_info:
            for type_id in type_ids:
                if type_id in cdb.addl_info['type_id2cuis']:
                    cui_filter.update(cdb.addl_info['type_id2cuis'][type_id])
                else:
                    raise Exception("Impossible to create filters, disable them.")
        else:
            raise Exception("Impossible to create filters, disable them.")

    return cui_filter

