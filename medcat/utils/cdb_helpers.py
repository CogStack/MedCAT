import pandas as pd

def umls_to_snomed_name_extension(mrconso_path, snomed_codes, column_names=None, sep='|', lng='ENG', output_path=None, use_umls_primary_names=False, **kwargs):
    r''' Prepare the MRCONSO.RRF to be used for extansion of SNOMED. Will output a CSV that can
    be used with cdb_maker (use the snomed_cdb.dat as the base one and extend with this).

    Args:
        mrconso_path (`str`):
            Path to the MRCONSO.RRF file from UMLS.
        snomed_codes (`Set[str]`):
            SNOMED codes that you want to extend with UMLS names.
        column_names (`str`, optional):
            Column names in the UMLS, leave blank and it will be autofiled.
        sep (`str`, defaults to `|`):
            Separator for the mrconso CSV (RRF is also a CSV)
        lng (`str`, defaults to `ENG`):
            What language to keep from the MRCONSO file
        output_path (`str`, optional):
            Where to save the built CSV - fullpath
        kwargs
            Will be forwarded to pandas.read_csv
        use_umls_primary_names (`bool`, defaults to False):
            If True the default names from UMLS will be used to inform medcat later once the CDB is built.
    Return:
        df (pandas.DataFrame):
            Dataframe with UMLS names and SNOMED CUIs.
    '''
    if column_names is None:
        column_names = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF', 'unk']
    df = pd.read_csv(mrconso_path, names=column_names, sep=sep, dtype=str, **kwargs)
    df = df[df.LAT == lng]

    umls2snomed = {}
    # Get all SNOMED terms
    df_snomed = df[df.SAB == 'SNOMEDCT_US']
    # Keep only the SNOMED Codes that we need
    df_snomed = df_snomed[[code in snomed_codes for code in df_snomed.CODE.values]]
    # Remove all CUIs that map to more than one SNOMED term
    cuis_to_remove = set()
    for pair in df_snomed[['CUI', 'CODE']].values:
        if pair[0] not in umls2snomed or pair[1] == umls2snomed[pair[0]]:
            umls2snomed[pair[0]] = pair[1]
        else:
            cuis_to_remove.add(pair[0])
    umls2snomed = {cui: snomed for cui, snomed in umls2snomed.items() if cui not in cuis_to_remove}
    # Keep only cui and str
    df = df[['CUI', 'STR', 'TTY']]
    # Replace UMLS with SNOMED codes
    df['CUI'] = [umls2snomed.get(cui, "unk-unk") for cui in df.CUI.values]
    df = df[df.CUI != 'unk-unk']

    # Change name status if required
    if use_umls_primary_names:
        df['TTY'] = ['P' if tty=='PN' else 'A' for tty in df['TTY'].values]
    else:
        df['TTY'] = ['A'] * len(df)

    # Remove name duplicates (keep TTY also in the eq)
    df = df.drop_duplicates(subset=['CUI', 'STR', 'TTY'])

    # Rename columns
    df.columns = ['cui', 'name', 'name_status']

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


def snomed_source_to_csv(snomed_term_paths=[], snomed_desc_paths=[], sep='\t', output_path=None, **kwargs):
    r''' Given paths to the snomed files with concepts e.g. `sct2_Concept_Snapshot_INT_20180731.txt` this will
    build a CSV required by the cdb_maker.py

    Args:
        snomed_term_paths (`List[str]`):
            One or many paths to the different `sct2_Concept_Snapshot_*` files
        snomed_desc_paths (`List[str]`):
            One or many paths to the different `sct2_Description_Snapshot_*` files
        sep (`str`, defaults to '\t'):
            The separator used in the snomed files.
        output_path (`str`, optional):
            Where to save the built CSV - fullpath
        kwargs:
            Will be forwarded to pandas.read_csv
    Return:
        snomed_cdb_df (pandas.DataFrame):
            Dataframe with SNOMED concepts ready to be used with medcat.cdb_maker.
    '''

    # Process terms
    snomed_terms = [pd.read_csv(path, sep=sep, dtype=str, **kwargs) for path in snomed_term_paths]
    snomed_terms = pd.concat(snomed_terms)
    snomed_terms = snomed_terms[snomed_terms.active == '1']

    # Process descs and keep only active ones (note this is not active concepts,
    #but active descriptions).
    snomed_descs = [pd.read_csv(path, sep=sep, dtype=str, **kwargs) for path in snomed_desc_paths]
    snomed_descs = pd.concat(snomed_descs)
    snomed_descs = snomed_descs[snomed_descs.active == '1']

    # Keep only active terms in the snomed_descs
    f = set(snomed_terms.id.values)
    snomed_descs = snomed_descs[[id in f for id in snomed_descs.conceptId.values]]

    # Remove everything that we do not need and rename columns
    snomed_cdb_df = snomed_descs[['conceptId', 'term', 'typeId']]
    snomed_cdb_df = snomed_cdb_df.rename(columns={"conceptId": "cui", "term": "name", 'typeId': 'name_status'})
    # Ontology is always SNOMED
    snomed_cdb_df['ontology'] = ['SNOMED'] * len(snomed_cdb_df)
    # Take primary names
    snomed_cdb_df['name_status'] = ['P' if name_status == '900000000000003001' else 'A' for name_status in snomed_cdb_df.name_status.values]

    # Add type_names and IDs, there is no real way to do this, so I'll invent a TUI
    cui2semantic_type_name = {cui:name[name.rfind("(")+1:name.rfind(")")] for cui, name in
            snomed_cdb_df[snomed_cdb_df['name_status'] == 'P'][['cui', 'name']].values if name.endswith(")")}
    # Create map from name2id
    semantic_type_name2id = {semantic_type_name: 'T-{}'.format(id) for id, semantic_type_name in enumerate(sorted(set(cui2semantic_type_name.values())))}

    # Add columns to the output df
    snomed_cdb_df['semantic_type_name'] = [cui2semantic_type_name.get(cui, 'unk') for cui in snomed_cdb_df.cui]
    snomed_cdb_df['semantic_type_id'] = [semantic_type_name2id.get(type_name, 'unk') for type_name in snomed_cdb_df.semantic_type_name]

    if output_path is not None:
        snomed_cdb_df.to_csv(output_path, index=False)

    return snomed_cdb_df
