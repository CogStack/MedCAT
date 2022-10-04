
import pandas as pd

_DEFAULT_COLUMNS: list = [
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",   
]

_DEFAULT_SEM_TYPE_COLUMNS: list = [
    "CUI",
    "TUI",
    "STN",
    "STY",
    "ATUI",
    "CVF",
]

medcat_csv_mapper: dict = {
    'CUI': 'cui',
    'STR': 'name',
    'SAB': 'ontologies',
    'ISPREF': 'name_status',
    'TUI': 'type_ids', # from MRSTY.RRF
}


class UMLS:
    r"""
    Pre-process UMLS release files:
    Args:
        main_file_name (str):
            Path to the main file name (probably MRCONSO.RRF)
        sem_types_file (str):
            Path to the semantic types file name (probably MRSTY.RRF)
        allow_langugages (list):
            Languages to filter out. Defaults to just English (['ENG']).
        sep (str):
            The separator used within the files. Defaults to '|'.
    """

    def __init__(self, main_file_name: str, sem_types_file: str, allow_languages: list = ['ENG'], sep: str = '|'):
        self.main_file_name = main_file_name
        self.sem_types_file = sem_types_file
        self.main_columns = list(_DEFAULT_COLUMNS) # copy
        self.sem_types_columns = list(_DEFAULT_SEM_TYPE_COLUMNS) # copy
        self.sep = sep
        # copy in case of default list
        self.allow_langugages = list(allow_languages) if allow_languages else allow_languages

    def to_concept_df(self) -> pd.DataFrame:
        """Create a concept DataFrame.
        The default column names are expected.

        Returns:
            pd.DataFrame: The resulting DataFrame
        """
        # target columns:
        # cui, name, name_status, ontologies, description_type_ids, type_ids
        df = pd.read_csv(self.main_file_name, names=self.main_columns, sep=self.sep, index_col=False)

        # filter languages
        if self.allow_langugages:
            df = df[df["LAT"].isin(self.allow_langugages)]

        # TODO filter by activity ?

        # get TUI

        sem_types = pd.read_csv(self.sem_types_file, names=self.sem_types_columns, sep=self.sep, index_col=False)
        df = df.merge(sem_types)

        # rename columns

        df = df.rename(columns=medcat_csv_mapper)

        # pop all unneccessary columns

        # all initial collumns should have been renamed
        for col_name in self.main_columns + self.sem_types_columns:
            if col_name in df.columns:
                df.pop(col_name)

        # looks like description_type_ids is not really used anywhere, so I won't look for it

        return df

    def map_umls2snomed(self) -> pd.DataFrame:
        """Map to SNOMED-CT.

        Currently, uses the SCUI column. At the time of writing, this is equal to the CODE column.
        But this may not be the case in the future.

        Returns:
            pd.DataFrame: Dataframe that contains the SCUI (source CUI) as well as the UMLS CUI for each applicable concept
        """
        df = pd.read_csv(self.main_file_name, names=self.main_columns, sep=self.sep, index_col=False, dtype={'SCUI': 'str'})
        # get only SNOMED-CT US based concepts that have a SNOMED-CT (source) CUI
        df = df[df.SAB == 'SNOMEDCT_US'][df.SCUI.notna()]
        # sort by SCUI
        df = df.sort_values(by='SCUI').reset_index(drop=True)
        # rearrange with SCUI as the first column
        df = df[['SCUI',] + [col for col in df.columns.values if col != 'SCUI']]
        return df


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Need to specify two file locations: MRCONSO.RRF and MRSTY.RRF')
        sys.exit(1)
    umls = UMLS(sys.argv[1], sys.argv[2])
    df = umls.to_concept_df()
    save_file = "preprocessed_umls.csv"
    print(f"Saving to {save_file}")
    df.to_csv(save_file, index=False)
    print('Converting to SNOMED')
    to_snomed = umls.map_umls2snomed()
    print('As SNOMED:')
    print(to_snomed.head())
