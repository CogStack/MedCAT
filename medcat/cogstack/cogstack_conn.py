import getpass
import elasticsearch
import elasticsearch.helpers
from IPython.display import display, HTML
from datetime import datetime

from typing import Dict, Optional


class CogStackConn(object):
    def __init__(self, host, port=9200, username: Optional[str] = None, password: Optional[str] = None, scheme: str='https',
                 timeout: int = 360, max_retries: int = 10, retry_on_timeout: bool = True, **kwargs):
        username, password = self._check_auth_details(username, password)

        self.elastic = elasticsearch.Elasticsearch(hosts=[{'host': host, 'port': port}],
                                         http_auth=(username, password),
                                         scheme=scheme,
                                         verify_certs=False,
                                         timeout=timeout,
                                         max_retries=max_retries,
                                         retry_on_timeout=retry_on_timeout,
                                         **kwargs)

    def _check_auth_details(self, username: Optional[str] = None, password: Optional[str] = None):
        if username is None:
            username = input("Username:")
        if password is None:
            password = getpass.getpass("Password:")

        # TODO: Implement auth check, for now I assume all is fine
        return username, password

    def get_docs_generator(self, query: Dict, index: str, es_gen_size: int=800, request_timeout: int=840000, **kwargs):
        docs_generator = elasticsearch.helpers.scan(self.elastic,
            query=query,
            index=index,
            size=es_gen_size,
            request_timeout=request_timeout,
            **kwargs)

        return docs_generator

    def get_text_for_doc(self, doc_id, index='epr_documents', text_field='body_analysed'):
        r = self.elastic.get(index=index, id=doc_id)
        text = r['_source'][text_field]
        return text

    def show_all_ent_cntx(self, stream, cui: str, cntx_size: int = 100, index='epr_documents', text_field='body_analysed'):
        for id in range(len(stream['entities'])):
            if stream['entities'][id]['conceptId'] == cui:
                print(stream['entities'][id]['name'])
                print("Status: " + stream['entities'][id]['metaSubject'])
                print("Presence: " + stream['entities'][id]['metaPresence'])
                print("Time: " + stream['entities'][id]['metaTime'])
                print("Date: " + str(datetime.fromtimestamp((stream['entities'][id]['timestamp']))))

                self.show_ent_cntx(stream, id, cntx_size, index, text_field)

    def show_ent_cntx(self, stream, id: int, cntx_size=100, index='epr_documents', text_field='body_analysed'):
        doc_id = stream['entities'][id]['documentId']
        text = self.get_text_for_doc(doc_id, index=index, text_field=text_field)

        start = stream['entities'][id]['start']
        c_start = max(0, start-cntx_size)
        end = stream['entities'][id]['end']
        c_end = min(len(text), end+cntx_size)

        ent_cntx = text[c_start:start] + "<span style='background-color: #53f725'>" + text[start:end] + "</span>" + text[end:c_end]
        ent_cntx.replace("\n", "<br />")
        display(HTML(ent_cntx))

        if len(text) < start:
            print("Text of the clincal note corrupted: " + text[0:100])


    def bulk_to_cogstack(self):
        # TODO: look the code made for Nazli/Dan
        pass
