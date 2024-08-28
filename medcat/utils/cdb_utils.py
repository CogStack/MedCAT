import logging
import numpy as np

from copy import deepcopy
from medcat.cdb import CDB

logger = logging.getLogger(__name__) # separate logger from the package-level one


def merge_cdb(cdb1: CDB, 
              cdb2: CDB, 
              overwrite_training: int = 0,
              full_build: bool = False) -> CDB:
    """Merge two CDB's together to produce a new, single CDB. The contents of inputs CDBs will not be changed.
    `addl_info` can not be perfectly merged, and will prioritise cdb1. see `full_build`

    Args:
        cdb1 (CDB):
            The first medcat cdb to merge. In cases where merging isn't suitable isn't ideal (such as
            cui2preferred_name), this cdb values will be prioritised over cdb2.
        cdb2 (CDB):
            The second medcat cdb to merge.
        overwrite_training (int):
            Choose to prioritise a CDB's context vectors values over merging gracefully. 0 - no prio, 1 - CDB1, 2 - CDB2
        full_build (bool):
            Add additional information from "addl_info" dicts "cui2ontologies" and "cui2description"

    Returns:
        CDB: The merged CDB.
    """
    config = deepcopy(cdb1.config)
    cdb = CDB(config)

    # Copy CDB 1 - as all settings from CDB 1 will be carried over
    cdb.cui2names = deepcopy(cdb1.cui2names)
    cdb.cui2snames = deepcopy(cdb1.cui2snames)
    cdb.cui2count_train = deepcopy(cdb1.cui2count_train)
    cdb.cui2info = deepcopy(cdb1.cui2info)
    cdb.cui2context_vectors = deepcopy(cdb1.cui2context_vectors)
    cdb.cui2tags = deepcopy(cdb1.cui2tags)
    cdb.cui2type_ids = deepcopy(cdb1.cui2type_ids)
    cdb.cui2preferred_name = deepcopy(cdb1.cui2preferred_name)
    cdb.name2cuis = deepcopy(cdb1.name2cuis)
    cdb.name2cuis2status = deepcopy(cdb1.name2cuis2status)
    cdb.name2count_train = deepcopy(cdb1.name2count_train)
    cdb.name_isupper = deepcopy(cdb1.name_isupper)
    if full_build:
        cdb.addl_info = deepcopy(cdb1.addl_info)

    # handles cui2names, cui2snames, name_isupper, name2cuis, name2cuis2status, cui2preferred_name
    for cui in cdb2.cui2names:
        names = dict()
        for name in cdb2.cui2names[cui]:
            names[name] = {'snames': cdb2.cui2snames.get(cui, set()), 'is_upper': cdb2.name_isupper.get(name, False), 'tokens': {}, 'raw_name': cdb2.get_name(cui)}
            name_status = cdb2.name2cuis2status.get(name, {}).get(cui, 'A') # get the name status if it exists, default to 'A'
        # For addl_info check cui2original_names as they MUST be added
        ontologies = set()
        description = ''
        to_build = False
        if full_build and (cui in cdb2.addl_info['cui2original_names'] or cui in cdb2.addl_info['cui2description']):
            to_build = True
            if 'cui2ontologies' in cdb2.addl_info:
                ontologies.update(cdb2.addl_info['cui2ontologies'][cui])
            if 'cui2description' in cdb2.addl_info:
                description = cdb2.addl_info['cui2description'][cui]
        cdb._add_concept(cui=cui, names=names, ontologies=ontologies, name_status=name_status,
                        type_ids=cdb2.cui2type_ids[cui], description=description, full_build=to_build)
        if cui in cdb1.cui2names:
            if (cui in cdb1.cui2count_train or cui in cdb2.cui2count_train) and not (overwrite_training == 1 and cui in cdb1.cui2count_train): 
                if overwrite_training == 2 and cui in cdb2.cui2count_train:
                    cdb.cui2count_train[cui] = cdb2.cui2count_train[cui]
                else:
                    cdb.cui2count_train[cui] = cdb1.cui2count_train.get(cui, 0) + cdb2.cui2count_train.get(cui, 0)
            if cui in cdb1.cui2context_vectors and not (overwrite_training == 1 and cui in cdb1.cui2context_vectors[cui]):
                if overwrite_training == 2 and cui in cdb2.cui2context_vectors:
                    weights = [0, 1]
                else:
                    norm = cdb.cui2count_train[cui]
                    weights = [np.divide(cdb1.cui2count_train.get(cui, 0), norm), np.divide(cdb2.cui2count_train.get(cui, 0), norm)]
                contexts = set(list(cdb1.cui2context_vectors.get(cui, {}).keys()) + list(cdb2.cui2context_vectors.get(cui, {}).keys())) # xlong, long, medium, short
                for s in contexts: 
                    cdb.cui2context_vectors[cui][s] = (weights[0] * cdb1.cui2context_vectors[cui].get(s, np.zeros(shape=(300)))) + (weights[1] * cdb2.cui2context_vectors[cui].get(s, np.zeros(shape=(300))))
            if cui in cdb1.cui2tags: 
                cdb.cui2tags[cui].append(cdb2.cui2tags[cui])
            if cui in cdb1.cui2type_ids: 
                cdb.cui2type_ids[cui] = cdb1.cui2type_ids[cui].union(cdb2.cui2type_ids[cui])
        else:
            if cui in cdb2.cui2count_train: 
                cdb.cui2count_train[cui] = cdb2.cui2names[cui]
            if cui in cdb2.cui2info: 
                cdb.cui2info[cui] = cdb2.cui2info[cui]
            if cui in cdb2.cui2context_vectors: 
                cdb.cui2context_vectors[cui] = cdb2.cui2context_vectors[cui]
            if cui in cdb2.cui2tags: 
                cdb.cui2tags[cui] = cdb2.cui2tags[cui]
            if cui in cdb2.cui2type_ids: 
                cdb.cui2type_ids[cui] = cdb2.cui2type_ids[cui]

    if overwrite_training != 1:
        for name in cdb2.name2cuis:
            if name in cdb1.name2cuis and overwrite_training == 0: # if they exist in both cdbs
                if name in cdb1.name2count_train and name in cdb2.name2count_train:
                    cdb.name2count_train[name] = str(int(cdb1.name2count_train[name]) + int(cdb2.name2count_train[name])) # these are strings for some reason
            else:
                if name in cdb2.name2count_train: 
                    cdb.name2count_train[name] = cdb2.name2count_train[name]

    # snames
    cdb.snames = cdb1.snames.union(cdb2.snames)

    # vocab, adding counts if they occur in both
    cdb.vocab = deepcopy(cdb1.vocab)
    if overwrite_training != 1:
        for word in cdb2.vocab:
            if word in cdb.vocab and overwrite_training == 0:
                cdb.vocab[word] += cdb2.vocab[word]
            else:
                cdb.vocab[word] = cdb2.vocab[word]

    return cdb
