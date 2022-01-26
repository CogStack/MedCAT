import os
import json
import re
import hashlib
import pandas as pd


def parse_file(filename, first_row_header=True, columns=None):
    with open(filename, encoding='utf-8') as f:
        entities = [[n.strip() for n in line.split('\t')] for line in f]
        return pd.DataFrame(entities[1:], columns=entities[0] if first_row_header else columns)


class Snomed:
    r"""
    Pre-process SNOMED CT release files:
    Args:
        data_path:
            Path to the unzipped SNOMED CT folder
        uk_ext: bool
            Specification of a SNOMED UK extension after 2021 to process the divergent release format.
    """

    def __init__(self, data_path, uk_ext=False):
        self.data_path = data_path
        self.release = data_path[-16:-8]
        self.uk_ext = uk_ext

    def to_concept_df(self, ):
        """
        Please remember to specify if the version is a SNOMED UK extension released after 2021.
        This can be done prior to this step: Snomed.uk_ext = True.
        This step is not required for UK extension releases pre-2021.

        :return: SNOMED CT concept DataFrame ready for MEDCAT CDB creation
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

            for f in os.listdir(contents_path):
                m = re.search(f'{concept_snapshot}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)

            int_terms = parse_file(f'{contents_path}/{concept_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_terms = int_terms[int_terms.active == '1']
            del int_terms

            int_desc = parse_file(f'{contents_path}/{description_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_descs = int_desc[int_desc.active == '1']
            del int_desc

            _ = pd.merge(active_terms, active_descs, left_on=['id'], right_on=['conceptId'], how='inner')
            del active_terms
            del active_descs

            active_with_primary_desc = _[_['typeId'] == '900000000000003001']  # active description
            active_with_synonym_desc = _[_['typeId'] == '900000000000013009']  # active synonym
            del _
            active_with_all_desc = pd.concat([active_with_primary_desc, active_with_synonym_desc])

            active_snomed_df = active_with_all_desc[['id_x', 'term', 'typeId']]
            del active_with_all_desc

            active_snomed_df = active_snomed_df.rename(columns={'id_x': 'cui', 'term': 'name', 'typeId': 'name_status'})
            active_snomed_df['ontologies'] = 'SNOMED-CT'
            active_snomed_df['name_status'] = active_snomed_df['name_status'].replace(
                ['900000000000003001', '900000000000013009'],
                ['P', 'A'])
            active_snomed_df = active_snomed_df.reset_index(drop=True)

            temp_df = active_snomed_df[active_snomed_df['name_status'] == 'P'][['cui', 'name']]
            temp_df['description_type_ids'] = temp_df['name'].str.extract(r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$")
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
        SNOMED CT provides a rich set of inter-relationships between concepts.

        :return: List of all SNOMED CT relationships
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

            for f in os.listdir(contents_path):
                m = re.search(f'{concept_snapshot}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            int_relat = parse_file(f'{contents_path}/{relationship_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_relat = int_relat[int_relat.active == '1']
            del int_relat

            all_rela.extend([relationship for relationship in active_relat["typeId"].unique()])
        return all_rela

    def relationship2json(self, relationshipcode, output_jsonfile):
        """

        :param relationshipcode: A single SCTID or unique concept identifier of the relationship type
        :param output_jsonfile: Name of json file output. Tip: include SNOMED edition to avoid downstream confusions
        :return: json file  of relationship mapping
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
            for f in os.listdir(contents_path):
                m = re.search(f'{concept_snapshot}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            int_relat = parse_file(f'{contents_path}/{relationship_snapshot}_{snomed_v}_{snomed_release}.txt')
            active_relat = int_relat[int_relat.active == '1']
            del int_relat

            relationship = dict([(key, []) for key in active_relat["destinationId"].unique()])
            for _, v in active_relat.iterrows():
                if v['typeId'] == str(relationshipcode):
                    _ = v['destinationId']
                    relationship[_].append(v['sourceId'])
                else:
                    pass
            output_dict.update(relationship)
        with open(output_jsonfile, 'w') as json_file:
            json.dump(output_dict, json_file)
        return

    def map_snomed2icd10(self):
        """

        :return: SNOMED to ICD10 mapping DataFrame which includes all metadata
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
        df2merge = []
        for i, snomed_release in enumerate(snomed_releases):
            refset_terminology = f'{paths[i]}/Snapshot/Refset/Map'
            icd10_ref_set = 'der2_iisssccRefset_ExtendedMapSnapshot'
            if self.uk_ext:
                if "SnomedCT_InternationalRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssccRefset_ExtendedMapSnapshot"
                elif "SnomedCT_UKClinicalRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssccRefset_ExtendedMapUKCLSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    icd10_ref_set = "der2_iisssccRefset_ExtendedMapUKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
            for f in os.listdir(refset_terminology):
                m = re.search(f'{icd10_ref_set}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            mappings = parse_file(f'{refset_terminology}/{icd10_ref_set}_{snomed_v}_{snomed_release}.txt')
            mappings = mappings[mappings.active == '1']
            icd_mappings = mappings.sort_values(by=['referencedComponentId', 'mapPriority', 'mapGroup']).reset_index(
                drop=True)
            df2merge.append(icd_mappings)
        return pd.concat(df2merge)

    def map_snomed2opcs4(self):
        """

        :return: SNOMED to OPSC4 mapping DataFrame which includes all metadata
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
        df2merge = []
        for i, snomed_release in enumerate(snomed_releases):
            refset_terminology = f'{paths[i]}/Snapshot/Refset/Map'
            snomed_v = ''
            opcs4_ref_set = 'der2_iisssccRefset_ExtendedMapSnapshot'
            if self.uk_ext:
                if "SnomedCT_InternationalRF2_PRODUCTION" in paths[i]:
                    continue
                elif "SnomedCT_UKClinicalRF2_PRODUCTION" in paths[i]:
                    opcs4_ref_set = "der2_iisssciRefset_ExtendedMapUKCLSnapshot"
                elif "SnomedCT_UKEditionRF2_PRODUCTION" in paths[i]:
                    opcs4_ref_set = "der2_iisssciRefset_ExtendedMapUKEDSnapshot"
                elif "SnomedCT_UKClinicalRefsetsRF2_PRODUCTION" in paths[i]:
                    continue
                else:
                    pass
            for f in os.listdir(refset_terminology):
                m = re.search(f'{opcs4_ref_set}'+r'_(.*)_\d*.txt', f)
                if m:
                    snomed_v = m.group(1)
            if snomed_v == '':
                raise FileNotFoundError("This SNOMED release does not contain OPCS mapping files")
            mappings = parse_file(f'{refset_terminology}/{opcs4_ref_set}_{snomed_v}_{snomed_release}.txt')
            mappings = mappings[mappings.active == '1']
            icd_mappings = mappings.sort_values(by=['referencedComponentId', 'mapPriority', 'mapGroup']).reset_index(
                drop=True)
            df2merge.append(icd_mappings)
        return pd.concat(df2merge)
