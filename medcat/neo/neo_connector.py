from py2neo import Graph
import getpass
from collections import defaultdict


class NeoConnector:
    def __init__(self, uri, user, password=None):
        if password is None:
            password = getpass.getpass("Password:")
        self.graph = Graph(uri, auth=(user, password))

    def execute(self, query):
        r = self.graph.run(query)
        return r

    def bucket_concepts(self, data, bucket_size_seconds):
        entities = data['entities']

        _bucket = []
        _concepts = set()
        start_time = -1
        new_stream = []
        # Sort entities
        entities.sort(key=lambda ent: ent['timestamp'])
        for ent in entities:
            if start_time == -1:
                start_time = ent['timestamp']

            if ent['timestamp'] - start_time >= bucket_size_seconds:
                # Add to stream
                new_stream.extend(_bucket)
                _bucket = []
                _concepts = set()
                start_time = ent['timestamp']

                t_ent = dict(new_stream[-1])
                t_ent['timestamp'] += 1
                t_ent['name'] = '<SEP>'
                t_ent['conceptId'] = '<SEP>'
                new_stream.append(t_ent)

            if ent['conceptId'] not in _concepts:
                _bucket.append(ent)
                _concepts.add(ent['conceptId'])

        if _bucket:
            new_stream.extend(_bucket)

        data['entities'] = new_stream

    def get_all_patients(self, concepts, limit=1000, require_time=False, ignore_meta=False):
        """Return all patients having all concepts

        Args:
            concepts: The concepts
            limit: The maximum number of results. Defaults to 1000.
            require_time: If set only concepts that have the timestamp property will be used.
        """

        q = "WITH [{}] AS cs ".format(",".join(["'{}'".format(c) for c in concepts]))
        if not require_time:
            q += '''MATCH (c:Concept)<-[:HAS '''
            if not ignore_meta:
                q += '''{metaPresence: 'True', metaSubject: 'Patient'}'''
            q += ''']-(:Document)<-[:HAS]-(pt:Patient)
            WHERE c.conceptId in cs
            WITH pt, size(cs) as inputCnt, count(DISTINCT c) as cnt
            WHERE cnt = inputCnt
            '''
        else:
            q += '''MATCH (c:Concept)<-[r:HAS {metaPresence: 'True', metaSubject:
            'Patient'}]-(:Document)<-[:HAS]-(pt:Patient) \n
            WHERE c.conceptId in cs AND exists(r.timestamp) \n
            WITH pt, size(cs) as inputCnt, count(DISTINCT c) as cnt \n
            WHERE cnt = inputCnt \n
            '''

        q += ' RETURN pt LIMIT {}'.format(limit)
        data = self.execute(q).data() # Do not like this too much 

        return [n['pt']['patientId'] for n in data], q

    def get_all_concepts_from(self, patient_id=None, document_id=None,
            limit=1000, bucket_size_seconds=None, min_count=0, meta_requirements=None, require_time=True):
        """Returns all concepts belonging to a document or patient
        given the concept type (if none all are retruned).
        """

        if patient_id is not None:
            q = 'MATCH (patient:Patient {patientId: "%s"})-[:HAS]->' % patient_id \
                + '(document:Document)-[has:HAS]->(concept:Concept) \n'
        elif document_id is not None:
            q = 'MATCH (patient:Patient)-[:HAS]->(document:Document {documentId: "%s"})' % document_id \
                + '-[has:HAS]->(concept:Concept) \n'
        else:
            raise Exception("patient_id or document_id are required")
        q += 'RETURN patient, document, concept, has LIMIT %s \n' % limit

        data = self.execute(q).data() # Do not like this too much 
        out = None
        if len(data) > 0:
            out = {'patient': dict(data[0]['patient']),
                   'entities': []}

            cnt = defaultdict(int)
            for row in data:
                if meta_requirements is None or \
                   all([row['has'][meta] == value for meta,value in meta_requirements.items()]):
                    if not require_time or 'timestamp' in row['has']:
                        ent = dict(row['concept']) # Take everything from concept
                        ent['documentId'] = row['document']['documentId']
                        ent.update(row['has']) # add all the stuff from the meta ann

                        out['entities'].append(ent)
                        cnt[ent['conceptId']] += 1

            # Cleanup based on min_count
            new_ents = []
            for ent in out['entities']:
                if cnt[ent['conceptId']] >= min_count:
                    ent['count'] = cnt[ent['conceptId']]
                    new_ents.append(ent)
            out['entities'] = new_ents

            if bucket_size_seconds is not None:
                self.bucket_concepts(data=out, bucket_size_seconds=bucket_size_seconds)

        return out, q

    def get_all_patients_descend(self, concepts, limit=1000, require_time=False):
        """Return all patients having all descendant concepts under the ancestor concept

        Args:
            concepts: Ancestor top-level concepts
            limit: The maximum number of results. Defaults to 1000.
            require_time: If set only concepts that have the timestamp property will be used.
                Defaults to False
        Returns:
            List: Patients with attached SNOMED concepts
        """

        q = "WITH [{}] AS ancestor ".format(",".join(["'{}'".format(c) for c in concepts]))
        if not require_time:
            q += '''MATCH (n:Concept)-[:IS_A*0..5]->(m:Concept)
                    WHERE m.conceptId IN ancestor ## get the ancestor and the children
                    WITH [n.conceptId] AS lineage ## pass the lineage to patient match
                    MATCH (c:Concept)<-[r:HAS {metaPresence: 'True', metaSubject: 'Patient'}]-(d:Document)<-[q:HAS]-(pt:Patient)
                    WHERE c.conceptId in lineage    
                    '''
        else:
            q += '''MATCH (n:Concept)-[:IS_A*0..5]->(m:Concept)
                    WHERE m.conceptId IN ancestor ## get the ancestor and the children
                    WITH [n.conceptId] AS lineage ## pass the lineage to patient match
                    MATCH (c:Concept)<-[r:HAS {metaPresence: 'True', metaSubject: 'Patient'}]-(d:Document)<-[q:HAS]-(pt:Patient)
                    WHERE c.conceptId in lineage AND exists(r.timestamp)
                    '''

        q += ' RETURN pt.patientId, pt.sex, c.conceptId, c.name, r.timestamp LIMIT {}'.format(limit)
        data = self.execute(q).data() # Do not like this too much 

        return [n['pt']['patientId'] for n in data], q
