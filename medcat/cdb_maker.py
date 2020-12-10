import pandas
import spacy
import numpy as np
import datetime
import logging
from functools import partial

from medcat.pipe import Pipe
from medcat.cdb import CDB
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.preprocessing.cleaners import prepare_name
from medcat.preprocessing.taggers import tag_skip_and_punct
from medcat.utils.loggers import add_handlers

class CDBMaker(object):
    r''' Given a CSV as shown in https://github.com/CogStack/MedCAT/tree/master/examples/<example> it creates a CDB or
    updates an exisitng one.

    Args:
        config (`medcat.config.Config`):
            Global config for MedCAT.
        cdb (`medcat.cdb.CDB`, optional):
            If set the `CDBMaker` will updat the existing `CDB` with new concepts in the CSV.
        name_max_words (`int`, defaults to `20`):
            Names with more words will be skipped during the build of a CDB
    '''
    log = logging.getLogger(__package__)
    log = add_handlers(log)
    def __init__(self, config, cdb=None, name_max_words=20):
        self.config = config
        # Set log level
        self.log.setLevel(self.config.general['log_level'])

        # To make life a bit easier
        self.cnf_cm = config.cdb_maker

        if cdb is None:
            self.cdb = CDB(config=self.config)
        else:
            self.cdb = cdb

        # Build the required spacy pipeline
        self.nlp = Pipe(tokenizer=spacy_split_all, config=config)
        self.nlp.add_tagger(tagger=partial(tag_skip_and_punct, config=self.config),
                            name='skip_and_punct',
                            additional_fields=['is_punct'])


    def prepare_csvs(self, csv_paths, sep=',', encoding=None, escapechar=None, index_col=False, full_build=False, only_existing_cuis=False, **kwargs):
        r''' Compile one or multipe CSVs into a CDB.

        Args:
            csv_paths (`List[str]`):
                An array of paths to the csv files that should be processed
            full_build (`bool`, defautls to `True`):
                If False only the core portions of the CDB will be built (the ones required for
                the functioning of MedCAT). If True, everything will be added to the CDB - this
                usually includes concept descriptions, various forms of names etc (take care that
                this option produces a much larger CDB).
            sep (`str`, defaults to `,`):
                If necessarya a custom separator for the csv files
            encoding (`str`, optional):
                Encoing to be used for reading the CSV file
            escapechar (`str`, optional):
                Escapechar for the CSV
            index_col (`bool`, defaults_to `False`):
                Index column for pandas read_csv
            only_existing_cuis (`bool`, defaults to False):
                If True no new CUIs will be added, but only linked names will be extended. Mainly used when
                enriching names of a CDB (e.g. SNOMED with UMLS terms).
        Return:
            `medcat.cdb.CDB` with the new concepts added.

        Note:
            **kwargs:
                Will be passed to pandas for CSV reading
            csv:
                Examples of the CSV used to make the CDB can be found on [GitHub](link)
        '''

        useful_columns = ['cui', 'name', 'ontologies', 'name_status', 'type_ids', 'description']
        name_status_options = {'A', 'P', 'N'}

        for csv_path in csv_paths:
            # Read CSV, everything is converted to strings
            df = pandas.read_csv(csv_path, sep=sep, encoding=encoding, escapechar=escapechar, index_col=index_col, dtype=str, **kwargs)
            df = df.fillna('')

            # Find which columns to use from the CSV
            cols = []
            col2ind = {}
            for col in list(df.columns):
                if str(col).lower().strip() in useful_columns:
                    col2ind[str(col).lower().strip()] = len(cols)
                    cols.append(col)

            self.log.info("Started importing concepts from: {}".format(csv_path))
            _time = None # Used to check speed
            _logging_freq = np.ceil(len(df[cols]) / 100)
            for row_id, row in enumerate(df[cols].values):
                if row_id % _logging_freq == 0:
                    # Print some stats
                    if _time is None:
                        # Add last time if it does not exist
                        _time = datetime.datetime.now()
                    # Get current time
                    ctime = datetime.datetime.now()
                    # Get time difference
                    timediff = ctime - _time
                    self.log.info("Current progress: {:.0f}% at {:.3f}s per {} rows".format(
                        (row_id / len(df)) * 100, timediff.microseconds/10**6 + timediff.seconds, (len(df[cols]) // 100)))
                    # Set previous time to current time
                    _time = ctime

                # This must exist
                cui = row[col2ind['cui']].strip().upper()

                if not only_existing_cuis or (only_existing_cuis and cui in self.cdb.cui2names):
                    if 'ontologies' in col2ind:
                        ontologies = set([ontology.strip() for ontology in row[col2ind['ontologies']].upper().split(self.cnf_cm['multi_separator']) if
                                         len(ontology.strip()) > 0])
                    else:
                        ontologies = set()

                    if 'name_status' in col2ind:
                        name_status = row[col2ind['name_status']].strip().upper()

                        # Must be allowed
                        if name_status not in name_status_options:
                            name_status = 'A'
                    else:
                        # Defaults to A - meaning automatic
                        name_status = 'A'

                    if 'type_ids' in col2ind:
                        type_ids = set([type_id.strip() for type_id in row[col2ind['type_ids']].upper().split(self.cnf_cm['multi_separator']) if
                                        len(type_id.strip()) > 0])
                    else:
                        type_ids = set()

                    # Get the ones that do not need any changing
                    if 'description' in col2ind:
                        description = row[col2ind['description']].strip()
                    else:
                        description = ""

                    # We can have multiple versions of a name
                    names = {} # {'name': {'tokens': [<str>], 'snames': [<str>]}}

                    raw_names = [raw_name.strip() for raw_name in row[col2ind['name']].split(self.cnf_cm['multi_separator']) if 
                                 len(raw_name.strip()) > 0]
                    for raw_name in raw_names:
                        raw_name = raw_name.strip()
                        prepare_name(raw_name, self.nlp, names, self.config)
                    self.cdb.add_concept(cui=cui, names=names, ontologies=ontologies, name_status=name_status, type_ids=type_ids,
                                         description=description, full_build=full_build)
                    # DEBUG
                    self.log.debug("\n\n**** Added\n CUI: {}\n Names: {}\n Ontologies: {}\n Name status: {}\n".format(cui, names, ontologies, name_status) + \
                                   " Type IDs: {}\n Description: {}\n Is full build: {}".format(
                                   type_ids, description, full_build))

        return self.cdb
