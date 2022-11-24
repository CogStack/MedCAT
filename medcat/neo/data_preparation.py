import os
import pandas as pd


def get_index_queries():
    """Run before everything to speed up things."""
    return ['CREATE INDEX patientId FOR (p:Patient) ON (p.patientId);',
            'CREATE INDEX conceptId FOR (c:Concept) ON (c.conceptId);',
            'CREATE INDEX documentId FOR (d:Document) ON (d.documentId);']


def create_neo_csv(data, columns, output_dir='/etc/lib/neo4j/import/',
                   base_name='patients'):
    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data.
        columns:
            What data to use from the dataframe.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible.
        base_name:
            Name of the csv.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.read_csv(data)

    # Remove duplicates
    df = df.drop_duplicates(subset=columns)

    out_df = df[columns]
    data_path = os.path.join(output_dir, f"{base_name}.csv")
    out_df.to_csv(data_path, index=False)


def create_patients_csv(data, output_dir='/etc/lib/neo4j/import/',
                        base_name='patients'):
    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data: patientId,
            sex, ethnicity, dob.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible,
            but writing there could be only admin.

    Returns:
        str: The query.
    """
    query = (
        'USING PERIODIC COMMIT 100000 \n'
        f'LOAD CSV WITH HEADERS FROM  "file:///{base_name}.csv" AS row \n'
        'CREATE (:Patient {patientId: toString(row.patientId), \n'
        '                  sex: toString(row.sex), \n'
        '                  ethnicity: toString(row.ethnicity), \n'
        '                  dob: datetime(row.dob)}) \n'
        )

    create_neo_csv(data=data, columns=['patientId', 'sex', 'ethnicity', 'dob'],
                   output_dir=output_dir, base_name=base_name)

    return query


def create_documents_csv(data, output_dir='/etc/lib/neo4j/import/',
                         base_name='documents'):
    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data: documentId.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible.

    Returns:
        str: The query.
    """
    query = (
        'USING PERIODIC COMMIT 100000 \n'
        f'LOAD CSV WITH HEADERS FROM  "file:///{base_name}.csv" AS row \n'
        'CREATE (:Document {documentId: toString(row.documentId)}) \n'
        )

    create_neo_csv(data=data, columns=['documentId'],
                   output_dir=output_dir, base_name=base_name)

    return query


def create_concepts_csv(data, output_dir='/etc/lib/neo4j/import/',
                         base_name='concepts'):
    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data: conceptId,
            name and type.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible.
    """
    query = (
        'USING PERIODIC COMMIT 100000 \n'
        f'LOAD CSV WITH HEADERS FROM  "file:///{base_name}.csv" AS row \n'
        'CREATE (:Concept {conceptId: toString(row.conceptId), \n'
        '                  type: toString(row.type), \n'
        '                  name: toString(row.name)}) \n'
        )

    create_neo_csv(data=data, columns=['conceptId', 'name', 'type'],
                   output_dir=output_dir, base_name=base_name)

    return query


def create_document2patient_csv(data, output_dir='/etc/lib/neo4j/import/',
                                base_name='document2patient'):

    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data: patientId and
            documentId.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible.
    """
    query = (
        'USING PERIODIC COMMIT 100000 \n'
        f'LOAD CSV WITH HEADERS FROM  "file:///{base_name}.csv" AS row \n'
        'MATCH (pt:Patient {patientId: toString(row.patientId)}) \n'
        'MATCH (doc:Document {documentId: toString(row.documentId)}) \n'
        'CREATE (pt)-[:HAS]->(doc); \n'
        )

    create_neo_csv(data=data, columns=['patientId', 'documentId'],
                   output_dir=output_dir, base_name=base_name)

    return query


def create_concept_ontology_csv(data, output_dir='/etc/lib/neo4j/import/',
                                base_name='concept_ontology'):

    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data: child, parent.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible.
    """
    query = (
        'USING PERIODIC COMMIT 100000 \n'
        f'LOAD CSV WITH HEADERS FROM  "file:///{base_name}.csv" AS row \n'
        'MATCH (child:Concept {conceptId: toString(row.child)}) \n'
        'MATCH (parent:Concept {conceptId: toString(row.parent)}) \n'
        'CREATE (child)-[:IS_A]->(parent); \n'
        )

    create_neo_csv(data=data, columns=['child', 'parent'],
                   output_dir=output_dir, base_name=base_name)

    return query


def create_document2concept_csv(data, output_dir='/etc/lib/neo4j/import/',
                         base_name='document2concepts'):
    """Creates a patients CSV for neo4j load csv function

    Args:
        data:
            A dataframe or path to a dataframe with the required data: 'conceptId',
            'documentId', 'contextSimilarity', 'start', 'end', 'timestamp',
            'metaSubject', 'metaPresence', 'metaTime'.
        output_dir:
            Where to save the CSVs, should be the neo4j imports path if possible.
    """
    query = (
        'USING PERIODIC COMMIT 100000 \n'
        f'LOAD CSV WITH HEADERS FROM  "file:///{base_name}.csv" AS row \n'
        'MATCH (doc:Document{documentId: toString(row.documentId)}) \n'
        'MATCH (concept:Concept {conceptId: toString(row.conceptId)}) \n'
        'CREATE (doc)-[:HAS {start: toInteger(row.start), \n'
        '                   end: toInteger(row.end), \n'
        '                   timestamp: toInteger(row.timestamp), \n'
        '                   contextSimilarity: toFloat(row.contextSimilarity), \n'
        '                   metaSubject: toString(row.metaSubject), \n'
        '                   metaPresence: toString(row.metaPresence), \n'
        '                   metaTime: toString(row.metaTime) \n'
        '            }]->(concept); \n'
        )

    columns = ['conceptId', 'documentId', 'contextSimilarity', 'start',
                'end', 'timestamp', 'metaSubject', 'metaPresence', 'metaTime']

    create_neo_csv(data=data, columns=columns,
                   output_dir=output_dir, base_name=base_name)

    return query


def get_data_from_docs(docs, doc2pt, doc2time=None):
    data = [['conceptId', 'documentId', 'contextSimilarity',
             'start', 'end', 'timestamp', 'metaSubject',
             'metaPresence', 'metaTime']]

    for doc_id, doc in docs.items():
        row = []
        for ent in doc['entities'].values():
            #if ent['meta_anns']['Subject']['value'] == 'Patient' and \
            #   ent['meta_anns']['Presence']['value'] == 'True':
            if doc2time is not None:
                t = doc2time[doc_id]
            else:
                t = ent['document_timestamp']

            row = [ent['cui'], doc_id,
                   ent['context_similarity'],
                   ent['start'], ent['end'],
                   t,
                   ent['meta_anns'].get('Subject', {}).get('value', None),
                   ent['meta_anns'].get('Presence', {}).get('value', None),
                   ent['meta_anns'].get('Time', {}).get('value', None)]
            data.append(row)
            row = []

    return data
