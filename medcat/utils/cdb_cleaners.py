import traceback

def clean_primary_name(cdb):
    # This will remove ambiguity for primary names 
    #as we assume that they are unique
    for cui in cdb.cui2pref_name.keys():
        name = cdb.cui2pref_name[cui]
        # Link only this cui to the pref_name, if name > 3 chars
        if len(str(name)) > 3:
            # Remove name from cuis
            cuis = list(cdb.name2cui[name])
            for _cui in cuis:
                if _cui != cui:
                    try:
                        cdb.cui2names[_cui].remove(name)
                    except Exception as e:
                        pass
            # Remove links apart from the choosen one
            cdb.name2cui[name] = {cui}


def clean_common_words(cdb, words):
    # This will remove words in words from
    #cdb
    # TODO: Here we need to cleanup more things, like the snames

    for word in words:
        # Remove word from CUIs
        if word in cdb.name2cui:
            cuis = list(cdb.name2cui[word])
            for cui in cuis:
                try:
                    cdb.cui2names[cui].remove(word)
                except Exception as e:
                    print(cui)
                    print(word)
                    print(str(e))
                    print()

                if len(cdb.cui2names[cui]) == 0:
                    del cdb.cui2names[cui]

            # Remove the word now
            del cdb.name2cui[word]
