import os
import json
import re
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


def parse_file(filename, first_row_header=True, columns=None):
    with open(filename, encoding='utf-8') as f:
        entities = [[n.strip() for n in line.split('\t')] for line in f]
        return pd.DataFrame(entities[1:], columns=entities[0] if first_row_header else columns)


def get_all_children(sctid, pt2ch):
    """
    Retrieves all the children of a given SNOMED CT ID (SCTID) from a given parent-to-child mapping (pt2ch) via the "IS A" relationship.
    pt2ch can be found in a MedCAT model in the additional info via the call: cat.cdb.addl_info['pt2ch']

    Args:
        sctid (int): The SCTID whose children need to be retrieved.
        pt2ch (dict): A dictionary containing the parent-to-child relationships in the form {parent_sctid: [list of child sctids]}.

    Returns:
        list: A list of unique SCTIDs that are children of the given SCTID.
    """
    result = []
    stack = [sctid]
    while len(stack) != 0:
        # remove the last element from the stack
        current_snomed = stack.pop()
        current_snomed_children = pt2ch.get(current_snomed, [])
        stack.extend(current_snomed_children)
        result.append(current_snomed)
    result = list(set(result))
    return result


def get_direct_refset_mapping(in_dict: dict) -> dict:
    """This method uses the output from Snomed.map_snomed2icd10 or
    Snomed.map_snomed2opcs4 and removes the metadata and maps each
    SNOMED CUI to the prioritised list of the target ontology CUIs.

    The input dict is expected to be in the following format:
    - Keys are SnomedCT CUIs
    - The values are lists of dictionaries, each list item (at least)
      - Has a key 'code' that specifies the target onotlogy CUI
      - Has a key 'mapPriority' that specifies the priority

    Args:
        in_dict (dict): The input dict.

    Returns:
        dict: The map from Snomed CUI to list of priorities list of target ontology CUIs.
    """
    ret_dict = dict()
    for k, vals in in_dict.items():
        # sort such that highest priority values are first
        svals = sorted(vals, key=lambda el: el['mapPriority'], reverse=True)
        # only keep the code / CUI
        ret_dict[k] = [v['code'] for v in svals]
    return ret_dict




_IGNORE_TAG = '##IGNORE-THIS##'


class RefSetFileType(Enum):
    concept = auto()
    description = auto()
    relationship = auto()
    refset = auto()


@dataclass
class FileFormatDescriptor:
    concept: str
    description: str
    relationship: str
    refset: str
    common_prefix: str = "sct2_"  # for concept, description, and relationship (but not refset)

    @classmethod
    def ignore_all(cls) -> 'FileFormatDescriptor':
        return cls(concept=_IGNORE_TAG, description=_IGNORE_TAG,
                   relationship=_IGNORE_TAG, refset=_IGNORE_TAG)

    def get_file_per_type(self, file_type: RefSetFileType) -> str:
        raw = self._get_raw(file_type)
        return raw if file_type == RefSetFileType.refset else self.common_prefix + raw

    def _get_raw(self, file_type: RefSetFileType) -> str:
        return getattr(self, file_type.name)

    def get_concept(self) -> str:
        return self.get_file_per_type(RefSetFileType.concept)

    def get_description(self) -> str:
        return self.get_file_per_type(RefSetFileType.description)

    def get_relationship(self) -> str:
        return self.get_file_per_type(RefSetFileType.relationship)

    def get_refset(self) -> str:
        return self.get_file_per_type(RefSetFileType.refset)


@dataclass
class ExtensionDescription:
    exp_name_in_folder: str
    exp_files: FileFormatDescriptor
    exp_2nd_part_in_folder: Optional[str] = None


# pattern has:                                       EXTENSION      PRODUCTION     RELEASE
SNOMED_FOLDER_NAME_PATTERN = re.compile("^SnomedCT_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d{8}T\d{6}Z$)")
PER_FILE_TYPE_PATHS = {
    RefSetFileType.concept: os.path.join("Snapshot", "Terminology"),
    RefSetFileType.description: os.path.join("Snapshot", "Terminology"),
    RefSetFileType.relationship: os.path.join("Snapshot", "Terminology"),
    RefSetFileType.refset: os.path.join("Snapshot", "Refset", "Map"),
}



class SupportedExtension(Enum):
    INTERNATIONAL = ExtensionDescription(
        exp_name_in_folder="InternationalRF2",
        exp_files=FileFormatDescriptor(
            concept="Concept_Snapshot",
            description="Description_Snapshot-en",
            relationship="Relationship_Snapshot",
            # NOTE: the below will be ignored for UK_CLIN bundle
            refset="der2_iisssccRefset_ExtendedMapSnapshot"
        ),
    )
    UK_CLINICAL = ExtensionDescription(
        exp_name_in_folder="UKClinicalRF2",
        exp_files=FileFormatDescriptor(
            concept="Concept_UKCLSnapshot",
            description="Description_UKCLSnapshot-en",
            relationship="Relationship_UKCLSnapshot",
            refset="der2_iisssciRefset_ExtendedMapUKCLSnapshot"
        ),
    )
    UK_CLINICAL_REFSET = ExtensionDescription(
        exp_name_in_folder="UKClinicalRefsetsRF2",
        exp_files=FileFormatDescriptor.ignore_all()
    )
    UK_EDITION = ExtensionDescription(
        exp_name_in_folder="UKEditionRF2",
        exp_files=FileFormatDescriptor(
            concept="Concept_UKEDSnapshot",
            description="Description_UKEDSnapshot-en",
            relationship="Relationship_UKEDSnapshot",
            refset="der2_iisssciRefset_ExtendedMapUKEDSnapshot"
        ),
    )
    UK_DRUG = ExtensionDescription(
        exp_name_in_folder="UKDrugRF2",
        exp_files=FileFormatDescriptor(
            concept="Concept_UKDGSnapshot",
            description="Description_UKDGSnapshot-en",
            relationship="Relationship_UKDGSnapshot",
            refset="der2_iisssciRefset_ExtendedMapUKDGSnapshot",
        ),
    )
    AU = ExtensionDescription(
        exp_name_in_folder="Release",
        exp_2nd_part_in_folder="AU1000036",
        exp_files=FileFormatDescriptor(
            concept="Concept_Snapshot",
            description="Description_Snapshot-en-AU",
            relationship="Relationship_Snapshot",
            refset=_IGNORE_TAG,
        ),
    )


@dataclass
class BundleDescriptor:
    extensions: List[SupportedExtension]
    ignores: Dict[RefSetFileType, List[SupportedExtension]] = field(default_factory=dict)

    def has_invalid(self, ext: SupportedExtension, file_types: Tuple[RefSetFileType]) -> bool:
        for ft in file_types:
            if ft not in self.ignores:
                continue
            exts2ignore = self.ignores[ft]
            if ext in exts2ignore:
                return True
        return False


class SupportedBundles(Enum):
    UK_CLIN = BundleDescriptor(
        extensions=[SupportedExtension.INTERNATIONAL, SupportedExtension.UK_CLINICAL,
                    SupportedExtension.UK_CLINICAL_REFSET, SupportedExtension.UK_EDITION],
        ignores={RefSetFileType.refset: [SupportedExtension.INTERNATIONAL]}
        )
    UK_DRUG_EXT = BundleDescriptor(
        extensions=[SupportedExtension.UK_DRUG, SupportedExtension.UK_EDITION],
        )


def match_partials_with_folders(exp_names: List[Tuple[str, Optional[str]]],
                                folder_names: List[str],
                                _group_nr1: int = 1, _group_nr2: int = 2) -> bool:
    if len(exp_names) > len(folder_names):
        return False
    available_folders = [os.path.basename(f) for f in folder_names]
    for exp_name, exp_name_p2 in exp_names:
        found_cur_name = False
        for fi, folder in enumerate(available_folders):
            m = SNOMED_FOLDER_NAME_PATTERN.match(folder)
            if not m:
                continue
            if m.group(_group_nr1) != exp_name:
                continue
            if exp_name_p2 and m.group(_group_nr2) != exp_name_p2:
                continue
            found_cur_name = True
            break
        if found_cur_name:
            available_folders.pop(fi)
        else:
            return False
    return True


class Snomed:
    """
    Pre-process SNOMED CT release files.

    This class is used to create a SNOMED CT concept DataFrame ready for MedCAT CDB creation.

    Attributes:
        data_path (str): Path to the unzipped SNOMED CT folder.
        release (str): Release of SNOMED CT folder.
        uk_ext (bool, optional): Specifies whether the version is a SNOMED UK extension released after 2021. Defaults to False.
        uk_drug_ext (bool, optional): Specifies whether the version is a SNOMED UK drug extension. Defaults to False.
        au_ext (bool, optional): Specifies whether the version is a AU release. Defaults to False.
    """
    NO_VERSION_DETECTED = 'N/A'

    def __init__(self, data_path):
        self.data_path = data_path
        self.bundle = self._determine_bundle(self.data_path)
        self.paths, self.snomed_releases, self.exts = self._check_path_and_release()

    @classmethod
    def _determine_bundle(cls, data_path) -> Optional[SupportedBundles]:
        if not os.path.exists(data_path) or not os.path.isdir(data_path):
            return None
        for bundle in SupportedBundles:
            folder_names = list(os.listdir(data_path))
            exp_names = [(ext.value.exp_name_in_folder, ext.value.exp_2nd_part_in_folder)
                         for ext in bundle.value.extensions]
            if match_partials_with_folders(exp_names, folder_names):
                return bundle
        return None

    def _set_extension(self, release: str, extension: SupportedExtension) -> None:
        # NOTE: now using the later refset IF by default
        # NOTE: the OPCS4 refset ID is only relevant for UK releases
        self.opcs_refset_id = '1382401000000109'
        if (extension in (SupportedExtension.UK_CLINICAL, SupportedExtension.UK_DRUG) and
                # using lexicographical comparison below
                # e.g "20240101" > "20231122" results in True
                # yet "20231121" > "20231122" results in False
                len(release) == len("20231122") and release < "20231122"):
            # NOTE for UK extensions starting from 20231122 the
            #      OPCS4 refset ID seems to be different
            self.opcs_refset_id = "1126441000000105"
        self._extension = extension

    @classmethod
    def _determine_extension(cls, folder_path: str,
                             _group_nr1: int = 1, _group_nr2: int = 2) -> SupportedExtension:
        folder_basename = os.path.basename(folder_path)
        m = SNOMED_FOLDER_NAME_PATTERN.match(folder_basename)
        if not m:
            raise UnkownSnomedReleaseException(
                f"Unable to determine extension for path {repr(folder_path)}. "
                f"Checking against pattern {SNOMED_FOLDER_NAME_PATTERN}")
        ext_str = m.group(_group_nr1)
        ext_str2 = m.group(_group_nr2)
        for extension in SupportedExtension:
            if extension.value.exp_name_in_folder != ext_str:
                continue
            if (extension.value.exp_2nd_part_in_folder and
                    extension.value.exp_2nd_part_in_folder != ext_str2):
                continue
            return extension
        ext_names_folders = ",".join([f"{ext.name} ({ext.value.exp_name_in_folder})"
                                      for ext in SupportedExtension])
        raise UnkownSnomedReleaseException(
            f"Cannot Find the extension for {folder_path}. "
            f"Tried the following extensions: {ext_names_folders}")

    @classmethod
    def _determine_release(cls, folder_path: str, strict: bool = True,
                           _group_nr: int = 3, _keep_chars: int = 8) -> str:
        folder_basename = os.path.basename(folder_path)
        match = SNOMED_FOLDER_NAME_PATTERN.match(folder_basename)
        if match is None and strict:
            raise UnkownSnomedReleaseException(f"No version found in '{folder_path}'")
        elif match is None:
            return cls.NO_VERSION_DETECTED
        return match.group(_group_nr)[:_keep_chars]

    def to_concept_df(self):
        """
        Create a SNOMED CT concept DataFrame.

        Creates a SNOMED CT concept DataFrame ready for MEDCAT CDB creation.
        Checks if the version is a UK extension release and sets the correct file names for the concept and description snapshots accordingly.
        Additionally, handles the divergent release format of the UK Drug Extension >v2021 with the `uk_drug_ext` variable.

        Returns:
            pandas.DataFrame: SNOMED CT concept DataFrame.
        """

        df2merge = []
        for i, snomed_release in enumerate(self.snomed_releases):
            self._set_extension(snomed_release, self.exts[i])
            contents_path = os.path.join(self.paths[i], PER_FILE_TYPE_PATHS[RefSetFileType.concept])
            concept_snapshot = self._extension.value.exp_files.get_concept()
            description_snapshot = self._extension.value.exp_files.get_description()
            if concept_snapshot is None or _IGNORE_TAG in concept_snapshot or (
                    self.bundle and self.bundle.value.has_invalid(
                        self._extension, [RefSetFileType.concept, RefSetFileType.description])):
                continue

            for f in os.listdir(contents_path):
                m = re.search(f'{concept_snapshot}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)

            int_terms = parse_file(
                f'{contents_path}/{concept_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_terms = int_terms[int_terms.active == '1']
            del int_terms

            int_desc = parse_file(
                f'{contents_path}/{description_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_descs = int_desc[int_desc.active == '1']
            del int_desc

            _ = pd.merge(active_terms, active_descs, left_on=[
                         'id'], right_on=['conceptId'], how='inner')
            del active_terms
            del active_descs

            active_with_primary_desc = _[
                _['typeId'] == '900000000000003001']  # active description
            active_with_synonym_desc = _[
                _['typeId'] == '900000000000013009']  # active synonym
            del _
            active_with_all_desc = pd.concat(
                [active_with_primary_desc, active_with_synonym_desc])

            active_snomed_df = active_with_all_desc[['id_x', 'term', 'typeId']]
            del active_with_all_desc

            active_snomed_df = active_snomed_df.rename(
                columns={'id_x': 'cui', 'term': 'name', 'typeId': 'name_status'})
            active_snomed_df['ontologies'] = 'SNOMED-CT'
            active_snomed_df['name_status'] = active_snomed_df['name_status'].replace(
                ['900000000000003001', '900000000000013009'],
                ['P', 'A'])
            active_snomed_df = active_snomed_df.reset_index(drop=True)

            temp_df = active_snomed_df[active_snomed_df['name_status'] == 'P'][[
                'cui', 'name']]
            temp_df['description_type_ids'] = temp_df['name'].str.extract(
                r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$")
            active_snomed_df = pd.merge(active_snomed_df, temp_df.loc[:, ['cui', 'description_type_ids']],
                                        on='cui',
                                        how='left')
            del temp_df

            # Hash semantic tag to get a 8 digit type_id code
            active_snomed_df['type_ids'] = active_snomed_df['description_type_ids'].apply(
                lambda x: int(hashlib.sha256(str(x).encode('utf-8')).hexdigest(), 16) % 10 ** 8)
            df2merge.append(active_snomed_df)

        return pd.concat(df2merge).reset_index(drop=True)

    def list_all_relationships(self):
        """
        List all SNOMED CT relationships.

        SNOMED CT provides a rich set of inter-relationships between concepts.

        Returns:
            list: List of all SNOMED CT relationships.
        """
        all_rela = []
        for i, snomed_release in enumerate(self.snomed_releases):
            self._set_extension(snomed_release, self.exts[i])
            contents_path = os.path.join(self.paths[i], PER_FILE_TYPE_PATHS[RefSetFileType.concept])
            concept_snapshot = self._extension.value.exp_files.get_concept()
            relationship_snapshot = self._extension.value.exp_files.get_relationship()
            if concept_snapshot is None or _IGNORE_TAG in concept_snapshot or (
                    self.bundle and self.bundle.value.has_invalid(
                        self._extension, [RefSetFileType.concept, RefSetFileType.description])):
                continue

            for f in os.listdir(contents_path):
                m = re.search(f'{concept_snapshot}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            int_relat = parse_file(
                f'{contents_path}/{relationship_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_relat = int_relat[int_relat.active == '1']
            del int_relat

            all_rela.extend(
                [relationship for relationship in active_relat["typeId"].unique()])
        return all_rela

    def relationship2json(self, relationshipcode, output_jsonfile):
        """
        Convert a single relationship map structure to JSON file.

        Args:
            relationshipcode (str): A single SCTID or unique concept identifier
                of the relationship type.
            output_jsonfile (str): Name of JSON file output.

        Returns:
            file: JSON file of relationship mapping.
        """
        output_dict = {}
        for i, snomed_release in enumerate(self.snomed_releases):
            self._set_extension(snomed_release, self.exts[i])
            contents_path = os.path.join(self.paths[i], PER_FILE_TYPE_PATHS[RefSetFileType.concept])
            concept_snapshot = self._extension.value.exp_files.get_concept()
            relationship_snapshot = self._extension.value.exp_files.get_relationship()
            if concept_snapshot is None or _IGNORE_TAG in concept_snapshot or (
                    self.bundle and self.bundle.value.has_invalid(
                        self._extension, [RefSetFileType.concept, RefSetFileType.description])):
                continue

            for f in os.listdir(contents_path):
                m = re.search(f'{concept_snapshot}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            int_relat = parse_file(
                f'{contents_path}/{relationship_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_relat = int_relat[int_relat.active == '1']
            del int_relat

            relationship = dict(
                [(key, []) for key in active_relat["destinationId"].unique()])
            for _, v in active_relat.iterrows():
                if v['typeId'] == str(relationshipcode):
                    _ = v['destinationId']
                    relationship[_].append(v['sourceId'])
                else:
                    pass
            output_dict = {key: output_dict.get(key, []) + relationship.get(key, []) for key in
                           set(list(output_dict.keys()) + list(relationship.keys()))}
        with open(output_jsonfile, 'w') as json_file:
            json.dump(output_dict, json_file)
        return

    def map_snomed2icd10(self):
        """
        This function maps SNOMED CT concepts to ICD-10 codes using the refset mappings provided in the SNOMED CT release package.

        Returns:
            dict: A dictionary containing the SNOMED CT to ICD-10 mappings including metadata.
        """
        snomed2icd10df = self._map_snomed2refset()
        return self._refset_df2dict(snomed2icd10df[0])

    def map_snomed2opcs4(self) -> dict:
        """
        This function maps SNOMED CT concepts to OPCS-4 codes using the refset mappings provided in the SNOMED CT release package.

        Then it calls the internal function _map_snomed2refset() to get the DataFrame containing the OPCS-4 mappings.
        The function then converts the DataFrame to a dictionary using the internal function _refset_df2dict()

        Raises:
            AttributeError: If OPCS-4 mappings aren't available.

        Returns:
            dict: A dictionary containing the SNOMED CT to OPCS-4 mappings including metadata.
        """
        if all(ext not in (SupportedExtension.UK_CLINICAL, SupportedExtension.UK_DRUG)
               for ext in self.exts):
            raise AttributeError(
                "OPCS-4 mapping does not exist in this edition")
        snomed2opcs4df = self._map_snomed2refset()[1]
        return self._refset_df2dict(snomed2opcs4df)

    def _check_path_and_release(self):
        """
        This function checks the path and release of the SNOMED CT data provided.
        It looks for the "Snapshot" folder within the data path, and if it's not found, it looks for any folder containing the name "SnomedCT".
        It then stores the path and release in separate lists.
        If no valid paths are found, it raises a FileNotFoundError.

        Returns:
            tuple: a tuple containing two lists, the first one is a list of the paths where the data is located and the second is a list of the releases of the data.

        Raises:
            FileNotFoundError: If the path to the SNOMED CT directory is incorrect.
        """
        snomed_releases = []
        paths = []
        exts = []
        if "Snapshot" in os.listdir(self.data_path):
            paths.append(self.data_path)
            snomed_releases.append(self._determine_release(self.data_path, strict=True))
            exts.append(self._determine_extension(self.data_path))
        else:
            for folder in os.listdir(self.data_path):
                if "SnomedCT" in folder:
                    paths.append(os.path.join(self.data_path, folder))
                    rel = self._determine_release(folder, strict=True)
                    snomed_releases.append(rel)
                    exts.append(self._determine_extension(paths[-1]))
        if len(paths) == 0:
            raise FileNotFoundError('Incorrect path to SNOMED CT directory')
        return paths, snomed_releases, exts

    def _refset_df2dict(self, refset_df: pd.DataFrame) -> dict:
        """
        This function takes a SNOMED refset DataFrame as an input and converts it into a dictionary.
        The DataFrame should contain the columns 'referencedComponentId','mapTarget','mapGroup','mapPriority','mapRule','mapAdvice'.

        Args:
            refset_df (pd.DataFrame) : DataFrame containing the refset data

        Returns:
            dict: mapping from SNOMED CT codes as key and the refset metadata list of dictionaries as values.
        """
        refset_dict = refset_df.groupby('referencedComponentId').apply(lambda group: [{'code': row['mapTarget'],
                                                                                       'mapGroup': row['mapPriority'],
                                                                                       'mapPriority': row['mapPriority'],
                                                                                       'mapRule': row['mapRule'],
                                                                                       'mapAdvice': row['mapAdvice']} for _, row in group.iterrows()]).to_dict()
        return refset_dict

    def _map_snomed2refset(self):
        """
        Maps SNOMED CT concepts to refset mappings provided in the SNOMED CT release package.

        This function maps SNOMED CT concepts using the refset mappings in the Snapshot/Refset/Map directory.
        The refset mappings can either be ICD-10 codes in international releases or OPCS4 codes for SNOMED UK_extension, if available.

        Returns:
            pd.DataFrame: Dataframe containing SNOMED CT to refset mappings and metadata.
            OR
            tuple: Tuple of dataframes containing SNOMED CT to refset mappings and metadata (ICD-10, OPCS4), if uk_ext is True.
        """
        dfs2merge = []
        for i, snomed_release in enumerate(self.snomed_releases):
            self._set_extension(snomed_release, self.exts[i])
            refset_terminology = os.path.join(self.paths[i], PER_FILE_TYPE_PATHS[RefSetFileType.refset])
            icd10_ref_set = self._extension.value.exp_files.get_refset()
            if icd10_ref_set is None or _IGNORE_TAG in icd10_ref_set or (
                    self.bundle and self.bundle.value.has_invalid(
                        self._extension, [RefSetFileType.concept, RefSetFileType.description])):
                continue
            for f in os.listdir(refset_terminology):
                m = re.search(f'{icd10_ref_set}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            mappings = parse_file(
                f'{refset_terminology}/{icd10_ref_set}_{snomed_v}_{snomed_release}.txt')
            mappings = mappings[mappings.active == '1']
            icd_mappings = mappings.sort_values(by=['referencedComponentId', 'mapPriority', 'mapGroup']).reset_index(
                drop=True)
            dfs2merge.append(icd_mappings)
        mapping_df = pd.concat(dfs2merge)
        del dfs2merge
        if any(ext in (SupportedExtension.UK_CLINICAL, SupportedExtension.UK_DRUG)
               for ext in self.exts):
            opcs_df = mapping_df[mapping_df['refsetId'] == self.opcs_refset_id]
            icd10_df = mapping_df[mapping_df['refsetId']
                                  == '999002271000000101']
            return icd10_df, opcs_df
        else:
            return mapping_df, None


class UnkownSnomedReleaseException(ValueError):

    def __init__(self, *args) -> None:
        super().__init__(*args)
