import traceback

def clean_primary_name(cdb):
    """ BE CAREFUL - I WOULD RECOMMEND NOT USING THIS FUNCTION
    """
    # This will remove ambiguity for primary names 
    #as we assume that they are unique
    for cui in cdb.cui2pref_name.keys():
        name = cdb.cui2pref_name[cui]
        # Link only this cui to the pref_name, if name > 4 chars
        if len(str(name)) > 4:
            # Remove name from cuis
            cuis = list(cdb.name2cui[name])
            for _cui in cuis:
                if _cui != cui:
                    skip = False
                    if _cui in cdb.cui2pref_name:
                        if name == cdb.cui2pref_name[_cui]:
                            skip = True
                            print("SKIP")
                            print(name)
                            print(_cui, cui)
                            print()
                    if not skip:
                        try:
                            cdb.cui2names[_cui].remove(name)
                            cdb.name2cui[name].remove(_cui)
                            print(name, _cui)
                            print()
                        except Exception as e:
                            print(e)
                            pass
            # Remove links apart from the choosen one
            cdb.name2cui[name] = {cui}


def clean_common_words(cdb, words):
    """ BE CAREFUL - I WOULD RECOMMEND NOT USING THIS FUNCTION
    """
    # This will mark common words as not unique
    for word in words:
        # Remove word from CUIs
        if word in cdb.name2cui:
            cuis = list(cdb.name2cui[word])
            for cui in cuis:
                try:
                    cdb.name_isunique[word] = False
                except Exception as e:
                    print(cui)
                    print(word)
                    print(str(e))
                    print()

                if len(cdb.cui2names[cui]) == 0:
                    del cdb.cui2names[cui]

            # Remove the word now
            del cdb.name2cui[word]

import re
def fix_snomed_names(cdb, cat):
    r = re.compile("\([^\(\)]+\)$")
    i = 0
    for cui in cdb.cui2pref_name.keys():
        i += 1
        name = cdb.cui2pretty_name[cui]
        name = r.sub("", name).strip()
        if len(name) > 3:
            cat.add_name(cui, name, is_pref_name=True)
        if i % 10000 == 0:
            print(i)


def fix_x_names(cdb, cat):
    i = 0
    for cui in cdb.cui2original_names:
        for name in list(cdb.cui2original_names[cui]):
            if name.startswith("[X]"):
                _name = name.replace("[X]", "")
                cat.add_name(cui, _name, is_pref_name=False, only_new=True)

                if i % 1000 == 0:
                    print(i)
                i += 1
