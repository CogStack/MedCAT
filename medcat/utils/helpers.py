import html
from medcat.cdb import CDB
from medcat.preprocessing.cleaners import clean_name
from medcat.utils.other import TPL_ENT, TPL_ENTS

from spacy import __version__ as spacy_version

import logging

logger = logging.getLogger(__name__)


def get_important_config_parameters(config):
    cnf = {
            "config.ner.min_name_len": {
                'value': config.ner.min_name_len,
                'description': "Minimum detection length (found terms/mentions shorter than this will not be detected)."
                },
            "config.ner.upper_case_limit_len": {
                'value': config.ner.upper_case_limit_len,
                'description': "All detected terms shorter than this value have to be uppercase, otherwise they will be ignored."
                },
            "config.linking.similarity_threshold": {
                'value': config.linking.similarity_threshold,
                'description': "If the confidence of the model is lower than this a detection will be ignore."
                },
            "config.linking.filters.cuis": {
                'value': len(config.linking.filters.cuis),
                'description': "Length of the CUIs filter to be included in outputs. If this is not 0 (i.e. not empty) its best to check what is included before using the model"
            },
            "config.general.spell_check": {
                'value': config.general.spell_check,
                'description': "Is spell checking enabled."
                },
            "config.general.spell_check_len_limit": {
                'value': config.general.spell_check_len_limit,
                'description': "Words shorter than this will not be spell checked."
                },
            }
    return cnf


def to_json_simple(docs, cdb):
    """
    output:  [{'text': <text>, 'entities': [<start,end,type>, ]}]
    """
    d = []

    for doc in docs:
        d.append({'text': doc.text, 'entities': [(e.start_char, e.end_char, cdb.tui2name[cdb.cui2tui[e.label_]]) for e in doc._.ents]})


def to_json_sumithra(docs, cdb):
    """
    output:  [
              [ text, {'entities': [<start,end,type>, ]} ],
              ...]
    """
    d = []

    for doc in docs:
        d.append([doc.text, {'entities': [(e.start_char, e.end_char, cdb.tui2name[cdb.cui2tui[e.label_]]) for e in doc._.ents]}])

    return d


def doc2html(doc):
    markup = ""
    offset = 0
    text = doc.text

    for span in list(doc.ents):
        start = span.start_char
        end = span.end_char
        fragments = text[offset:start].split("\n")

        for i, fragment in enumerate(fragments):
            markup += html.escape(fragment)
            if len(fragments) > 1 and i != len(fragments) - 1:
                markup += "</br>"
        ent = {'label': '', 'id': span._.id, 'bg': "rgb(74, 154, 239, {})".format(span._.context_similarity * span._.context_similarity + 0.12), 'text': html.escape(span.text)}
        # Add the entity
        markup += TPL_ENT.format(**ent)
        offset = end
    markup += html.escape(text[offset:])

    out = TPL_ENTS.format(content=markup, dir='ltr')

    return out


def json2html(doc):
    markup = ""
    offset = 0
    text = doc['text']

    for span in list(doc['entities']):
        start = span['start']
        end = span['end']
        fragments = text[offset:start].split("\n")

        for i, fragment in enumerate(fragments):
            markup += html.escape(fragment)
            if len(fragments) > 1 and i != len(fragments) - 1:
                markup += "</br>"
        ent = {'label': '', 'id': span['id'], 'bg': "rgb(74, 154, 239, {})".format(span._.context_similarity * span._.context_similarity + 0.12), 'text': html.escape(span['str'])}
        # Add the entity
        markup += TPL_ENT.format(**ent)
        offset = end
    markup += html.escape(text[offset:])

    out = TPL_ENTS.format(content=markup, dir='ltr')

    return out


def prepare_name(cat, name, version='CLEAN'):
    """Cleans up the name."""
    name = clean_name(name)

    if version.lower() == 'clean':
        sc_name = cat(name)
        tokens = [str(t.lemma_).lower() for t in sc_name if not t._.is_punct
                  and not t._.to_skip]

    if version.lower() == 'raw':
        sc_name = cat(name)
        tokens = [t.lower_ for t in sc_name if not t._.is_punct
                  and not (t._.to_skip and not t.is_stop)]

    if version.lower() == 'none':
        sc_name = cat(name)
        tokens = [t.lower_ for t in sc_name]


    # Join everything and return name
    name = "".join(tokens)
    return name, tokens


def get_all_from_name(name, nlp, source_value, SEP="", version='clean'):
    sc_name = nlp(source_value)
    name, tokens = prepare_name(nlp, name=name, version=version)
    tokens_vocab = [t.lower_ for t in sc_name if not t._.is_punct]

    snames = []
    sname = ""
    for token in tokens:
        sname = sname + token + SEP
        snames.append(sname.strip())

    return name, tokens, snames, tokens_vocab


def tkn_inds_from_doc(spacy_doc, text_inds=None, source_val=None):
    tkn_inds = None
    start = None
    end = None
    if text_inds is None and source_val in spacy_doc.text:
        start = spacy_doc.text.index(source_val)
        end = start + len(source_val)
    elif text_inds is not None:
        start = text_inds[0]
        end = text_inds[1]

    if start is not None:
        tkn_inds = []
        for tkn in spacy_doc:
            if tkn.idx >= start and tkn.idx <= end:
                tkn_inds.append(tkn.i)

    return tkn_inds


def tkns_from_doc(spacy_doc, start, end):
    tkns = []
    for tkn in spacy_doc:
        if tkn.idx >= start and tkn.idx <= end:
            tkns.append(tkn)

    return tkns


def filter_cdb_by_icd10(cdb: CDB) -> CDB:
    """Filters an existing CDB to only contain concepts that have an associated ICD-10 code.
    Can be used for snomed orr UMLS CDBs.

    Args:
        CDB: The input CDB

    Returns:
        CDB: The filtered CDB
    """
    cuis_to_keep = [cui for cui in cdb.cui2names.keys() if 'icd10' in cdb.cui2info[cui]]
    cdb.filter_by_cui(cuis_to_keep)
    return cdb


def umls_to_icd10cm(cdb, csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        try:
            cuis = str(row['CUI']).split("|")
            chapter = row['Class ID'].split('/')[-1]
            name = row['Preferred Label']
            for cui in cuis:
                if cui is not None and cui in cdb.cui2names:
                    icd10 = {'chapter': chapter, 'name': name}

                    if 'icd10' in cdb.cui2info[cui]:
                        # Check is the chapter already in
                        isin = False
                        for tmp in cdb.cui2info[cui]['icd10']:
                            if tmp['chapter'] == chapter:
                                isin = True
                        if not isin:
                            cdb.cui2info[cui]['icd10'].append(icd10)
                    else:
                        cdb.cui2info[cui]["icd10"] = [icd10]
        except Exception as e:
            logger.warn("Issue at %s", row["CUI"], exc_info=e)


def umls_to_icd10_over_snomed(cdb, pickle_path):
    import pickle
    u2i = pickle.load(open(pickle_path, 'rb'))

    for cui in u2i.keys():
        if cui in cdb.cui2names:
            if cui not in cdb.cui2info:
                cdb.cui2info[cui] = {}

            for icd10 in u2i[cui]:
                if 'icd10' in cdb.cui2info[cui]:
                    # If it exists skip it
                    pass
                else:
                    logger.info("%s %s", cui, icd10)
                    cdb.cui2info[cui]['icd10'] = [icd10]
            else:
                pass


def umls_to_icd10_ext(cdb, pickle_path):
    import pickle
    u2i = pickle.load(open(pickle_path, 'rb'))

    for cui in u2i.keys():
        if cui in cdb.cui2names:
            if cui in cdb.cui2info and 'icd10' not in cdb.cui2info[cui]:
                icd10 = u2i[cui]

                logger.info("%s %s", cui, icd10)
                cdb.cui2info[cui]['icd10'] = [icd10]


def umls_to_icd10(cdb, csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        try:
            cui = str(row['cui'])
            chapter = row['chapter']
            name = row['name']

            if cui is not None and cui in cdb.cui2names:
                icd10 = {'chapter': chapter, 'name': name}

                if 'icd10' in cdb.cui2info[cui]:
                    # Check is the chapter already in
                    isin = False
                    for tmp in cdb.cui2info[cui]['icd10']:
                        if tmp['chapter'] == chapter:
                            isin = True
                    if not isin:
                        cdb.cui2info[cui]['icd10'].append(icd10)
                else:
                    cdb.cui2info[cui]["icd10"] = [icd10]
        except Exception as e:
            logger.warn("Issue at %s", row["CUI"], exc_info=e)


def umls_to_snomed(cdb, pickle_path):
    """Map UMLS CDB to SNOMED concepts."""
    import pickle

    data = pickle.load(open(pickle_path, 'rb'))

    for key in data.keys():
        cui = str(key)
        for snomed_cui in data[key]:
            if "S-" in str(snomed_cui):
                snomed_cui = str(snomed_cui)
            else:
                snomed_cui = "S-" + str(snomed_cui)

            if key in cdb.cui2info:
                if 'snomed' in cdb.cui2info[key]:
                    cdb.cui2info[cui]['snomed'].append(snomed_cui)
                else:
                    cdb.cui2info[cui]['snomed'] = [snomed_cui]


def snomed_to_umls(cdb, pickle_path):
    """Map SNOMED CDB to UMLS concepts."""
    import pickle

    data = pickle.load(open(pickle_path, 'rb'))

    for key in data.keys():
        for umls_cui in data[key]:
            # Add S if it is not there
            if "S-" in str(key):
                cui = key
            else:
                cui = "S-" + str(key)

            if cui in cdb.cui2info:
                if 'umls' in cdb.cui2info[cui]:
                    cdb.cui2info[cui]['umls'].append(umls_cui)
                else:
                    cdb.cui2info[cui]['umls'] = [umls_cui]


def snomed_to_icd10(cdb, csv_path):
    """Add map from cui to icd10 for concepts."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        icd = str(row['icd10'])
        name = str(row['name'])

        if "S-" in str(row['cui']):
            cui = str(row['cui'])
        else:
            cui = "S-" + str(row['cui'])


        if cui in cdb.cui2names and icd is not None and icd != 'nan' and len(icd) > 0:
            icd = {'chapter': icd, 'name': name}

            if 'icd10' in cdb.cui2info[cui]:
                cdb.cui2info[cui]['icd10'].append(icd)
            else:
                cdb.cui2info[cui]['icd10'] = [icd]


def snomed_to_desc(cdb, csv_path):
    """Add descriptions to the concepts."""
    import pandas as pd
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        desc = row['desc']

        if "S-" in str(row['cui']):
            cui = str(row['cui'])
        else:
            cui = "S-" + str(row['cui'])


        # Check do we have this concept at all
        if cui in cdb.cui2names:
            # If yes add description
            if cui not in cdb.cui2desc:
                cdb.cui2desc[cui] = str(desc)
            elif str(desc) not in str(cdb.cui2desc[cui]):
                cdb.cui2desc[cui] = str(cdb.cui2desc[cui]) + "\n\n" + str(desc)


def filter_only_icd10(doc, cat):
    ents = []
    for ent in doc._.ents:
        if 'icd10' in cat.cdb.cui2info.get(ent._.cui, {}):
            ents.append(ent)
    doc._.ents = ents
    doc.ents = []
    cat.spacy_cat._create_main_ann(doc)


def add_names_icd10(csv_path, cat):
    import pandas as pd
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        try:
            cui = str(row['cui'])
            name = row['name']
            cat.add_name(cui, name, is_pref_name=False, only_new=True)
        except Exception as e:
            logger.warn("Issue at %s", row["CUI"], exc_info=e)

        if index % 1000 == 0:
            logger.info('index=%d', index)


def add_names_icd10cm(cdb, csv_path, cat):
    import pandas as pd
    df = pd.read_csv(csv_path)

    for index, row in df.iterrows():
        try:
            cuis = str(row['CUI']).split("|")
            name = row['Preferred Label']
            for cui in cuis:
                bl = len(cdb.cui2names.get(cui, []))
                cat.add_name(cui, name, is_pref_name=False, only_new=True)
                if bl != len(cdb.cui2names.get(cui, [])):
                    logger.info("'%s' with cui '%s'", name, cui)
        except Exception as e:
            logger.warn("Issue at %s", row["CUI"], exc_info=e)
            break

        if index % 1000 == 0:
            logger.info("Index=%d", index)


def remove_icd10_ranges(cdb):
    for cui in cdb.cui2info:
        if 'icd10' in cdb.cui2info[cui]:
            new_icd = []
            for icd in list(cdb.cui2info[cui]['icd10']):
                if '-' not in icd['chapter']:
                    new_icd.append(icd)
            if len(new_icd) > 0:
                cdb.cui2info[cui]['icd10'] = new_icd
            else:
                del cdb.cui2info[cui]['icd10']


def dep_check_scispacy():
    # IGNORE FUNCTION
    import spacy
    import subprocess
    import sys
    try:
        _ = spacy.load("en_core_sci_md")
    except Exception:
        logger.info("Installing the missing models for scispacy\n")
        pkg = 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz'
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])


def run_cv(cdb_path, data_path, vocab_path, cv=100, nepochs=16, test_size=0.1, lr=1, groups=None, **kwargs):
    from medcat.cat import CAT
    from medcat.utils.vocab import Vocab
    import json

    use_groups = False
    if groups is not None:
        use_groups = True

    f1s = {}
    ps = {}
    rs = {}
    tps = {}
    fns = {}
    fps = {}
    cui_counts = {}
    examples = {}
    for _ in range(cv):
        cdb = CDB()
        cdb.load_dict(cdb_path)
        vocab = Vocab()
        vocab.load_dict(path=vocab_path)
        # This does not conform to the latest API which requires config
        cat = CAT(cdb, vocab=vocab)
        cat.train = False
        cat.spacy_cat.MIN_ACC = 0.30
        cat.spacy_cat.MIN_ACC_TH = 0.30

        # Add groups if they exist
        if groups is not None:
            for cui in cdb.cui2info.keys():
                if "group" in cdb.cui2info[cui]:
                    del cdb.cui2info[cui]['group']
            groups = json.load(open("./groups.json"))
            for k,v in groups.items():
                for val in v:
                    cat.add_cui_to_group(val, k)

        # cat.train_supervised does not accept lr
        fp, fn, tp, p, r, f1, cui_counts, examples = cat.train_supervised(data_path=data_path,
                             lr=1, test_size=test_size, use_groups=use_groups, nepochs=nepochs, **kwargs)

        for key in f1.keys():
            if key in f1s:
                f1s[key].append(f1[key])
            else:
                f1s[key] = [f1[key]]

            if key in ps:
                ps[key].append(p[key])
            else:
                ps[key] = [p[key]]

            if key in rs:
                rs[key].append(r[key])
            else:
                rs[key] = [r[key]]

            if key in tps:
                tps[key].append(tp.get(key, 0))
            else:
                tps[key] = [tp.get(key, 0)]

            if key in fps:
                fps[key].append(fp.get(key, 0))
            else:
                fps[key] = [fp.get(key, 0)]

            if key in fns:
                fns[key].append(fn.get(key, 0))
            else:
                fns[key] = [fn.get(key, 0)]

    return fps, fns, tps, ps, rs, f1s, cui_counts, examples


def has_new_spacy() -> bool:
    """Figures out whether or not a newer version of spacy is installed.

    This plays a role in how some parts of the Span needs to be interacted with.

    As of writing, the new version starts at v3.3.1.

    Returns:
        bool: Whether new version was detected.
    """
    major, minor, patch_plus = spacy_version.split('.')
    major, minor = int(major), int(minor)
    patch = int(patch_plus)
    return (major > 3 or
            (major == 3 and minor > 3) or
            (major == 3 and minor == 3 and patch >= 1))
