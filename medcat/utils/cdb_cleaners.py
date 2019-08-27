def clean_primary_name(cdb):
    # This will remove ambiguity for primary names 
    #as we assume that they are unique

    for cui in cdb.cui2pref_name.keys():
        name = cdb.cui2pref_name[cui]
        # Link only this cui to the pref_name
        cdb.name2cui[name] = {cui}
