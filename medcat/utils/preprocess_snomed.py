import os
import json
import re
import hashlib
import pandas as pd
from typing import Optional, Dict, List
from collections import defaultdict
from enum import Enum


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


class SnapshotData:
    def __init__(self,
                 concept_snapshots: Dict[str, Optional[str]],
                 description_snapshots: Dict[str, Optional[str]],
                 relationship_snapshots: Dict[str, Optional[str]],
                 refset_snapshots: Dict[str, Optional[str]],
                 avoids: List[str] = ["UKClinicalRefsetsRF2_PRODUCTION"],
                 common_key_prefix: str = "SnomedCT_",
                 common_val_prefix: str = "sct2_", # NOT for refset snapshot
                 ):
        self.concept_snapshots = concept_snapshots
        self.description_snapshots = description_snapshots
        self.relationship_snapshots = relationship_snapshots
        self.refset_snapshots = refset_snapshots
        self.avoids = avoids
        self.common_key_prefix = common_key_prefix
        self.common_val_prefix = common_val_prefix

    def get_appropriate_name(self, part: Dict[str, Optional[str]], cur_path: str,
                             use_val_prefix: bool = True) -> Optional[str]:
        val_prefix = self.common_val_prefix if use_val_prefix else ''
        try:
            return val_prefix + part[cur_path]
        except KeyError:
            pass
        for raw_avoid in self.avoids:
            avoid = self.common_key_prefix + raw_avoid
            if avoid in cur_path:
                return None
        for raw_k, v in part.items():
            k = self.common_key_prefix + raw_k
            if k in cur_path:
                return val_prefix + v
        return None


class SupportedExtensions(Enum):
    INTERNATIONAL = SnapshotData(
        defaultdict(lambda: "Concept_Snapshot"),
        defaultdict(lambda: "Description_Snapshot-en"),
        defaultdict(lambda: "Relationship_Snapshot"),
        defaultdict(lambda: "der2_iisssccRefset_ExtendedMapSnapshot")
    )
    UK = SnapshotData(
        {
            "InternationalRF2_PRODUCTION": "Concept_Snapshot",
            "UKClinicalRF2_PRODUCTION": "Concept_UKCLSnapshot",
            "UKEditionRF2_PRODUCTION": "Concept_UKEDSnapshot",
        },
        {
            "InternationalRF2_PRODUCTION": "Description_Snapshot-en",
            "UKClinicalRF2_PRODUCTION": "Description_UKCLSnapshot-en",
            "UKEditionRF2_PRODUCTION": "Description_UKEDSnapshot-en",
        },
        {
            "InternationalRF2_PRODUCTION": "Relationship_Snapshot",
            "UKClinicalRF2_PRODUCTION": "Relationship_UKCLSnapshot",
            "UKEditionRF2_PRODUCTION": "Relationship_UKEDSnapshot",
        },
        {
            "InternationalRF2_PRODUCTION": None, # avoid
            "UKClinicalRF2_PRODUCTION": "der2_iisssciRefset_ExtendedMapUKCLSnapshot",
            "UKEditionRF2_PRODUCTION": "der2_iisssciRefset_ExtendedMapUKEDSnapshot",
        }
    )
    UK_DRUG = SnapshotData(
        {
            "UKDrugRF2_PRODUCTION": "Concept_UKDGSnapshot",
            "UKEditionRF2_PRODUCTION": "Concept_UKEDSnapshot",
        },
        {
            "UKDrugRF2_PRODUCTION": "Description_UKDGSnapshot-en",
            "UKEditionRF2_PRODUCTION": "Description_UKEDSnapshot-en",
        },
        {
            "InternationalRF2_PRODUCTION": "Relationship_Snapshot",
            "UKDrugRF2_PRODUCTION": "Relationship_UKDGSnapshot",
            "UKEditionRF2_PRODUCTION": "Relationship_UKEDSnapshot",
        },
        {
            "UKDrugRF2_PRODUCTION": "der2_iisssciRefset_ExtendedMapUKDGSnapshot",
            "UKEditionRF2_PRODUCTION": "der2_iisssciRefset_ExtendedMapUKEDSnapshot",
        }
    )
    AU = SnapshotData(
        defaultdict(lambda: "Concept_Snapshot"),
        defaultdict(lambda: "Description_Snapshot-en-AU"),
        defaultdict(lambda: "Relationship_Snapshot"),
        defaultdict(lambda: "der2_iisssccRefset_ExtendedMapSnapshot")
    )

    def get_concept_snapshot(self, cur_path: str) -> Optional[str]:
        return self.value.get_appropriate_name(self.value.concept_snapshots, cur_path)

    def get_description_snapshot(self, cur_path: str) -> Optional[str]:
        return self.value.get_appropriate_name(self.value.description_snapshots, cur_path)

    def get_relationship_snapshot(self, cur_path: str) -> Optional[str]:
        return self.value.get_appropriate_name(self.value.relationship_snapshots, cur_path)

    def get_refset_terminology(self, cur_path: str) -> Optional[str]:
        return self.value.get_appropriate_name(self.value.refset_snapshots, cur_path,
                                               use_val_prefix=False)


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
    SNOMED_RELEASE_PATTERN = re.compile("^SnomedCT_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d{8}T\d{6}Z$)")
    NO_VERSION_DETECTED = 'N/A'

    def __init__(self, data_path):
        self.data_path = data_path
        self.paths, self.snomed_releases, self.exts = self._check_path_and_release()

    def _set_extension(self, release: str, extension: SupportedExtensions) -> None:
        self.opcs_refset_id = "1126441000000105"
        if (extension in (SupportedExtensions.UK, SupportedExtensions.UK_DRUG) and
                # using lexicographical comparison below
                # e.g "20240101" > "20231122" results in True
                # yet "20231121" > "20231122" results in False
                len(release) == len("20231122") and release >= "20231122"):
            # NOTE for UK extensions starting from 20231122 the
            #      OPCS4 refset ID seems to be different
            self.opcs_refset_id = '1382401000000109'
        self._extension = extension

    @classmethod
    def _determine_extension(cls, folder_path: str) -> SupportedExtensions:
        uk_ext = "SnomedCT_UK" in folder_path
        uk_drug_ext = uk_ext and "Drug" in folder_path
        au_ext = "_AU" in folder_path
        # validate
        if (uk_ext or uk_drug_ext) and au_ext:
            raise UnkownSnomedReleaseException(
                "Cannot both be a UK and and a AU version. "
                f"Got UK={uk_ext}, UK_Drug={uk_drug_ext}, AU={au_ext}")
        if uk_drug_ext:
            return SupportedExtensions.UK_DRUG
        elif uk_ext:
            return SupportedExtensions.UK
        elif au_ext:
            return SupportedExtensions.AU
        return SupportedExtensions.INTERNATIONAL

    @classmethod
    def _determine_release(cls, folder_path: str, strict: bool = True,
                           _group_nr: int = 3, _keep_chars: int = 8) -> str:
        folder_basename = os.path.basename(folder_path)
        match = cls.SNOMED_RELEASE_PATTERN.match(folder_basename)
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
            contents_path = os.path.join(self.paths[i], "Snapshot", "Terminology")
            concept_snapshot = self._extension.get_concept_snapshot(self.paths[i])
            description_snapshot = self._extension.get_description_snapshot(self.paths[i])
            if concept_snapshot is None:
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
            contents_path = os.path.join(self.paths[i], "Snapshot", "Terminology")
            concept_snapshot = self._extension.get_concept_snapshot(self.paths[i])
            relationship_snapshot = self._extension.get_relationship_snapshot(self.paths[i])
            if concept_snapshot is None:
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
            contents_path = os.path.join(self.paths[i], "Snapshot", "Terminology")
            concept_snapshot = self._extension.get_concept_snapshot(self.paths[i])
            relationship_snapshot = self._extension.get_relationship_snapshot(self.paths[i])
            if concept_snapshot is None:
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
        if self._extension in (SupportedExtensions.UK, SupportedExtensions.UK_DRUG):
            return self._refset_df2dict(snomed2icd10df[0])
        else:
            return self._refset_df2dict(snomed2icd10df)

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
        if self._extension not in (SupportedExtensions.UK, SupportedExtensions.UK_DRUG):
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
            refset_terminology = f'{self.paths[i]}/Snapshot/Refset/Map'
            icd10_ref_set = self._extension.get_refset_terminology(self.paths[i])
            if icd10_ref_set is None:
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
        if self._extension in (SupportedExtensions.UK, SupportedExtensions.UK_DRUG):
            opcs_df = mapping_df[mapping_df['refsetId'] == self.opcs_refset_id]
            icd10_df = mapping_df[mapping_df['refsetId']
                                  == '999002271000000101']
            return icd10_df, opcs_df
        else:
            return mapping_df


class UnkownSnomedReleaseException(ValueError):

    def __init__(self, *args) -> None:
        super().__init__(*args)
