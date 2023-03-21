
from typing import List, Union
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
    """Pre-process UMLS release files:
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

    def map_umls2icd10(self) -> pd.DataFrame:
        """Map to ICD-10.

        Available SAB's that contain 'ICD10':
         - CCSR_ICD10CM - CCSR_ICD10CM (Clinical Classifications Software Refined for ICD-10-CM) - Synopsis
         - CCSR_ICD10PCS - CCSR_ICD10PCS (Clinical Classifications Software Refined for ICD-10-PCS) - Synopsis
         - DMDICD10 - DMDICD10 (ICD-10 German) - Statistics
         - ICD10AE - ICD10AE (ICD-10, American English Equivalents) - Synopsis
         - ICD10AMAE - ICD10AMAE (ICD-10, Australian Modification, Americanized English Equivalents) - Synopsis
         - ICD10AM - ICD10AM (ICD-10, Australian Modification) - Synopsis
         - ICD10DUT - ICD10DUT (ICD10, Dutch Translation) - Synopsis
         - ICD10PCS - ICD10PCS (ICD-10 Procedure Coding System) - Synopsis
         - ICD10 - ICD10 (International Classification of Diseases and Related Health Problems, Tenth Revision) - Synopsis
         - ICPC2ICD10DUT - ICPC2ICD10DUT (ICPC2-ICD10 Thesaurus, Dutch Translation) - Synopsis
         - ICPC2ICD10ENG - ICPC2ICD10ENG (ICPC2-ICD10 Thesaurus) - Synopsis
         - MTHICPC2ICD10AE - MTHICPC2ICD10AE (ICPC2E-ICD10 Thesaurus, American English Equivalents) - Synopsis

        Currently only using 'ICD10'. But others may be relevant as well.

        If one wants to use one of the other sources listed above,
        they would need to use the map_umls2source method.

        Returns:
            pd.DataFrame: DataFrame that has the ICD-10 codes
        """
        return self.map_umls2source(sources='ICD10')

    def map_umls2source(self, sources: Union[str, List[str]]) -> pd.DataFrame:
        """Allows mapping to an arbitrary

        Args:
            sources (Union[str, List[str]]): The source or sources to include.

        Returns:
            pd.DataFrame: DataFrame that has the target source codes
        """
        df = pd.read_csv(self.main_file_name, names=self.main_columns, sep=self.sep, index_col=False, dtype={'CODE': 'str'})
        # get the specified source(s)
        if isinstance(sources, list):
            df = df[df.SAB.isin(sources)][df.CODE.notna()]
        else:
            df = df[df.SAB == sources][df.CODE.notna()]
        # sort by CODE
        df = df.sort_values(by='CODE').reset_index(drop=True)
        # rearrange columns starting with CODE
        df = df[['CODE',] + [col for col in df.columns.values if col != 'CODE']]
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
    to_ICD10 = umls.map_umls2icd10()
    print('As ICD-10:')
    print(to_ICD10.head())
    to_ICD10_man = umls.map_umls2source(sources=['ICD10'])
    print('As ICD-10(MAN):')
    print(to_ICD10_man.head())
