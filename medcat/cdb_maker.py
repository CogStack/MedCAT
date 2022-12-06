import pandas as pd
import numpy as np
import datetime
import logging
import re
from typing import Optional, List, Dict, Union

from medcat.pipe import Pipe
from medcat.cdb import CDB
from medcat.config import Config
from medcat.preprocessing.tokenizers import spacy_split_all
from medcat.preprocessing.cleaners import prepare_name
from medcat.preprocessing.taggers import tag_skip_and_punct

PH_REMOVE = re.compile("(\s)\([a-zA-Z]+[^\)\(]*\)($)")


logger = logging.getLogger(__name__)


class CDBMaker(object):
    """Given a CSV as shown in https://github.com/CogStack/MedCAT/tree/master/examples/<example> it creates a CDB or
    updates an exisitng one.

    Args:
        config (medcat.config.Config):
            Global config for MedCAT.
        cdb (medcat.cdb.CDB):
            If set the `CDBMaker` will updat the existing `CDB` with
            new concepts in the CSV (Default value `None`).
    """

    def __init__(self, config: Config, cdb: Optional[CDB] = None) -> None:
        self.config = config
        # Set log level
        logger.setLevel(self.config.general['log_level'])

        # To make life a bit easier
        self.cnf_cm = config.cdb_maker

        if cdb is None:
            self.cdb = CDB(config=self.config)
        else:
            self.cdb = cdb

        # Build the required spacy pipeline
        self.pipe = Pipe(tokenizer=spacy_split_all, config=config)
        self.pipe.add_tagger(tagger=tag_skip_and_punct,
                             name='skip_and_punct',
                             additional_fields=['is_punct'])

    def prepare_csvs(self,
                     csv_paths: Union[pd.DataFrame, List[str]],
                     sep: str = ',',
                     encoding: Optional[str] = None,
                     escapechar: Optional[str] = None,
                     index_col: bool = False,
                     full_build: bool = False,
                     only_existing_cuis: bool = False, **kwargs) -> CDB:
        r"""Compile one or multiple CSVs into a CDB.

        Args:
            csv_paths (Union[pd.DataFrame, List[str]]):
                An array of paths to the csv files that should be processed. Can also be an array of pd.DataFrames
            full_build (bool):
                If False only the core portions of the CDB will be built (the ones required for
                the functioning of MedCAT). If True, everything will be added to the CDB - this
                usually includes concept descriptions, various forms of names etc (take care that
                this option produces a much larger CDB) (Default value False).
            sep (str):
                If necessary a custom separator for the csv files (Default value ',').
            encoding (str):
                Encoding to be used for reading the CSV file (Default value `None`).
            escapechar (str):
                Escape char for the CSV (Default value None).
            index_col (bool):
                Index column for pandas read_csv (Default value False).
            only_existing_cuis bool):
                If True no new CUIs will be added, but only linked names will be extended. Mainly used when
                enriching names of a CDB (e.g. SNOMED with UMLS terms) (Default value `False`).
        Returns:
            medcat.cdb.CDB: CDB with the new concepts added.

        Note:
            \*\*kwargs:
                Will be passed to pandas for CSV reading
            csv:
                Examples of the CSV used to make the CDB can be found on [GitHub](link)
        """

        useful_columns = ['cui', 'name', 'ontologies', 'name_status', 'type_ids', 'description']
        name_status_options = {'A', 'P', 'N'}

        for csv_path in csv_paths:
            # Read CSV, everything is converted to strings
            if isinstance(csv_path, str):
                logger.info("Started importing concepts from: {}".format(csv_path))
                df = pd.pandas.read_csv(csv_path, sep=sep, encoding=encoding, escapechar=escapechar, index_col=index_col, dtype=str, **kwargs)
            else:
                # Not very clear, but csv_path can be a pre-loaded csv
                df = csv_path
            df = df.fillna('')

            # Find which columns to use from the CSV
            cols: List = []
            col2ind = {}
            for col in list(df.columns):
                if str(col).lower().strip() in useful_columns:
                    col2ind[str(col).lower().strip()] = len(cols)
                    cols.append(col)

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
                    logger.info("Current progress: {:.0f}% at {:.3f}s per {} rows".format(
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
                    names: Dict = {} # {'name': {'tokens': [<str>], 'snames': [<str>]}}

                    raw_names = [raw_name.strip() for raw_name in row[col2ind['name']].split(self.cnf_cm['multi_separator']) if
                                 len(raw_name.strip()) > 0]
                    for raw_name in raw_names:
                        raw_name = raw_name.strip()
                        prepare_name(raw_name, self.pipe.spacy_nlp, names, self.config)

                        if self.config.cdb_maker.get('remove_parenthesis', 0) > 0 and name_status == 'P':
                            # Should we remove the content in parenthesis from primary names and add them also
                            raw_name = PH_REMOVE.sub(" ", raw_name).strip()
                            if len(raw_name) >= self.config.cdb_maker['remove_parenthesis']:
                                prepare_name(raw_name, self.pipe.spacy_nlp, names, self.config)

                    self.cdb.add_concept(cui=cui, names=names, ontologies=ontologies, name_status=name_status, type_ids=type_ids,
                                         description=description, full_build=full_build)
                    # DEBUG
                    logger.debug("\n\n**** Added\n CUI: %s\n Names: %s\n Ontologies: %s\n Name status: %s\n Type IDs: %s\n Description: %s\n Is full build: %s",
                                   cui, names, ontologies, name_status, type_ids, description, full_build)

        return self.cdb

    def destroy_pipe(self) -> None:
        self.pipe.destroy()
