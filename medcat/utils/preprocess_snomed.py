import os
import json
import pandas as pd
import hashlib
import re


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
        extension (optional):
            Is a SNOMED CT extension release file set
            # TODO: allow for extensions

    """

    def __init__(self, data_path, extension=False):
        self.data_path = data_path
        self.extension = extension
        self.release = data_path[-16:-8]

    def to_concept_df(self):
        """

        :return: SNOMED CT concept DataFrame
        """
        contents_path = os.path.join(self.data_path, "Snapshot", "Terminology")
        for f in os.listdir(contents_path):
            m = re.search(r'sct2_Concept_Snapshot_(.*)_\d*.txt', f)
            if m:
                snomed_v = m.group(1)

        int_terms = parse_file(f'{contents_path}/sct2_Concept_Snapshot_{snomed_v}_{self.release}.txt')
        active_terms = int_terms[int_terms.active == '1']
        del int_terms

        int_desc = parse_file(f'{contents_path}/sct2_Description_Snapshot-en_{snomed_v}_{self.release}.txt')
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

        active_snomed_df.rename(columns={'id_x': 'cui', 'term': 'name', 'typeId': 'name_status'}, inplace=True)
        active_snomed_df['ontologies'] = 'SNOMED-CT'
        active_snomed_df['name_status'] = active_snomed_df['name_status'].replace(
            ['900000000000003001', '900000000000013009'],
            ['P', 'A'])
        active_snomed_df.reset_index(drop=True, inplace=True)

        temp_df = active_snomed_df[active_snomed_df['name_status'] == 'P'][['cui', 'name']]
        temp_df['description_type_ids'] = temp_df['name'].str.extract(r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$")
        active_snomed_df = pd.merge(active_snomed_df, temp_df[['cui', 'description_type_ids']], on='cui', how='left')
        del temp_df

        # Hash semantic tag to get a 8 digit type_id code
        active_snomed_df['type_ids'] = active_snomed_df['description_type_ids'].apply(
            lambda x: int(hashlib.sha256(x.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

        return active_snomed_df

    def list_all_relationships(self):
        """
        SNOMED CT provides a rich set of inter-relationships between concepts.

        :return: List of all SNOMED CT relationships
        """
        contents_path = os.path.join(self.data_path, "Snapshot", "Terminology")
        for f in os.listdir(contents_path):
            m = re.search(r'sct2_Concept_Snapshot_(.*)_\d*.txt', f)
            if m:
                snomed_v = m.group(1)
        int_relat = parse_file(f'{contents_path}/sct2_Relationship_Snapshot_{snomed_v}_{self.release}.txt')
        active_relat = int_relat[int_relat.active == '1']
        del int_relat

        all_rela = [relationship for relationship in active_relat["typeId"].unique()]
        return all_rela

    def relationship2json(self, relationshipcode, output_jsonfile):
        """

        :param relationshipcode: The SCTID or unique concept identifier of the relationship type
        :param output_jsonfile: Name of json file output. Tip: include SNOMED edition
        :return: json file  of relationship mapping
        """
        contents_path = os.path.join(self.data_path, "Snapshot", "Terminology")
        for f in os.listdir(contents_path):
            m = re.search(r'sct2_Concept_Snapshot_(.*)_\d*.txt', f)
            if m:
                snomed_v = m.group(1)
        int_relat = parse_file(f'{contents_path}/sct2_Relationship_Snapshot_{snomed_v}_{self.release}.txt')
        active_relat = int_relat[int_relat.active == '1']
        del int_relat

        relationship = dict([(key, []) for key in active_relat["destinationId"].unique()])
        for index, v in active_relat.iterrows():
            if v['typeId'] == str(relationshipcode):
                _ = v['destinationId']
                relationship[_].append(v['sourceId'])
            else:
                pass
        with open(output_jsonfile, 'w') as json_file:
            json.dump(relationship, json_file)
        return

    def map_snomed2icd10(self):
        """

        :return: SNOMED to ICD10 mapping DataFrame
        """
        refset_terminology = f'{self.data_path}/Snapshot/Refset/Map'
        for f in os.listdir(contents_path):
            m = re.search(r'sct2_Concept_Snapshot_(.*)_\d*.txt', f)
            if m:
                snomed_v = m.group(1)
        mappings = parse_file(f'{refset_terminology}/der2_iisssccRefset_ExtendedMapSnapshot_{snomed_v}_{self.release}.txt')
        mappings = mappings[mappings.active == '1']
        icd_mappings = mappings.sort_values(by=['referencedComponentId', 'mapPriority', 'mapGroup']).reset_index(
            drop=True)

        return icd_mappings



