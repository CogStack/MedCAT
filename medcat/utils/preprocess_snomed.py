import os
import json
import re
import hashlib
import pandas as pd


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


class Snomed:
    """
    Pre-process SNOMED CT release files.

    This class is used to create a SNOMED CT concept DataFrame ready for MedCAT CDB creation.

    Attributes:
        data_path (str): Path to the unzipped SNOMED CT folder.
        release (str): Release of SNOMED CT folder.
        uk_ext (bool, optional): Specifies whether the version is a SNOMED UK extension released after 2021. Defaults to False.
        uk_drug_ext (bool, optional): Specifies whether the version is a SNOMED UK drug extension. Defaults to False.
    """

    def __init__(self, data_path, uk_ext=False, uk_drug_ext=False):
        self.data_path = data_path
        self.release = data_path[-16:-8]
        self.uk_ext = uk_ext
        self.uk_drug_ext = uk_drug_ext

    def to_concept_df(self):
        """
        Create a SNOMED CT concept DataFrame.

        Creates a SNOMED CT concept DataFrame ready for MEDCAT CDB creation.
        Checks if the version is a UK extension release and sets the correct file names for the concept and description snapshots accordingly.
        Additionally, handles the divergent release format of the UK Drug Extension >v2021 with the `uk_drug_ext` variable.

        Returns:
            pandas.DataFrame: SNOMED CT concept DataFrame.
        """
        paths, snomed_releases = self._check_path_and_release()

        df2merge = []
        for i, snomed_release in enumerate(snomed_releases):
            contents_path = os.path.join(paths[i], "Snapshot", "Terminology")
            concept_snapshot = "sct2_Concept_Snapshot"
            description_snapshot = "sct2_Description_Snapshot-en"
            if self.uk_ext:
                if "SnomedCT_UKClinicalRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKCLSnapshot"
                    description_snapshot = "sct2_Description_UKCLSnapshot-en"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKEDSnapshot"
                    description_snapshot = "sct2_Description_UKEDSnapshot-en"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
            if self.uk_drug_ext:
                if "SnomedCT_UKDrugRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKDGSnapshot"
                    description_snapshot = "sct2_Description_UKDGSnapshot-en"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKEDSnapshot"
                    description_snapshot = "sct2_Description_UKEDSnapshot-en"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass

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
        paths, snomed_releases = self._check_path_and_release()
        all_rela = []
        for i, snomed_release in enumerate(snomed_releases):
            contents_path = os.path.join(paths[i], "Snapshot", "Terminology")
            concept_snapshot = "sct2_Concept_Snapshot"
            relationship_snapshot = "sct2_Relationship_Snapshot"
            if self.uk_ext:
                if "SnomedCT_InternationalRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_Snapshot"
                    relationship_snapshot = "sct2_Relationship_Snapshot"
                elif "SnomedCT_UKClinicalRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKCLSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKCLSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKEDSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
            if self.uk_drug_ext:
                if "SnomedCT_UKDrugRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKDGSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKDGSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKEDSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass

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
        paths, snomed_releases = self._check_path_and_release()
        output_dict = {}
        for i, snomed_release in enumerate(snomed_releases):
            contents_path = os.path.join(paths[i], "Snapshot", "Terminology")
            concept_snapshot = "sct2_Concept_Snapshot"
            relationship_snapshot = "sct2_Relationship_Snapshot"
            if self.uk_ext:
                if "SnomedCT_InternationalRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_Snapshot"
                    relationship_snapshot = "sct2_Relationship_Snapshot"
                elif "SnomedCT_UKClinicalRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKCLSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKCLSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKEDSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
            if self.uk_drug_ext:
                if "SnomedCT_UKDrugRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKDGSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKDGSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    concept_snapshot = "sct2_Concept_UKEDSnapshot"
                    relationship_snapshot = "sct2_Relationship_UKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass

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
        if self.uk_ext is True:
            return self._refset_df2dict(snomed2icd10df[0])
        else:
            return self._refset_df2dict(snomed2icd10df)

    def map_snomed2opcs4(self):
        """
        This function maps SNOMED CT concepts to OPCS-4 codes using the refset mappings provided in the SNOMED CT release package.

        Then it calls the internal function _map_snomed2refset() to get the DataFrame containing the OPCS-4 mappings.
        The function then converts the DataFrame to a dictionary using the internal function _refset_df2dict()

        Returns:
            dict: A dictionary containing the SNOMED CT to OPCS-4 mappings including metadata.
        """
        if self.uk_ext is not True:
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
        if "Snapshot" in os.listdir(self.data_path):
            paths.append(self.data_path)
            snomed_releases.append(self.release)
        else:
            for folder in os.listdir(self.data_path):
                if "SnomedCT" in folder:
                    paths.append(os.path.join(self.data_path, folder))
                    snomed_releases.append(folder[-16:-8])
        if len(paths) == 0:
            raise FileNotFoundError('Incorrect path to SNOMED CT directory')
        return paths, snomed_releases

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

        Raises:
            FileNotFoundError: If the path to the SNOMED CT directory is incorrect.
        """
        paths, snomed_releases = self._check_path_and_release()
        dfs2merge = []
        for i, snomed_release in enumerate(snomed_releases):
            refset_terminology = f'{paths[i]}/Snapshot/Refset/Map'
            icd10_ref_set = 'der2_iisssccRefset_ExtendedMapSnapshot'
            if self.uk_ext:
                if "SnomedCT_InternationalRF2_PRODUCTION" in paths[i]:
                    continue
                elif "SnomedCT_UKClinicalRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssciRefset_ExtendedMapUKCLSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssciRefset_ExtendedMapUKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
            if self.uk_drug_ext:
                if "SnomedCT_UKDrugRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssciRefset_ExtendedMapUKDGSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssciRefset_ExtendedMapUKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
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
        if self.uk_ext or self.uk_drug_ext:
            opcs_df = mapping_df[mapping_df['refsetId'] == '1126441000000105']
            icd10_df = mapping_df[mapping_df['refsetId']
                                  == '999002271000000101']
            return icd10_df, opcs_df
        else:
            return mapping_df
