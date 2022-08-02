import getpass
import elasticsearch
import elasticsearch.helpers
import eland as ed
import pandas as pd
from tqdm.autonotebook import tqdm
from IPython.display import display, HTML
from datetime import datetime

from typing import Dict, List, Optional, Union


class CogStackConn(object):
    def __init__(self, hosts, username, password,
                 api_username, api_password, api: bool = False,
                 timeout: int = 360, max_retries: int = 10, retry_on_timeout: bool = True, **kwargs):
        if api:
            api_username, api_password = self._check_api_auth_details(api_username, api_password)
            self.elastic = elasticsearch.Elasticsearch(hosts=hosts,
                                                       api_key=(api_username, api_password),
                                                       verify_certs=False,
                                                       timeout=timeout,
                                                       max_retries=max_retries,
                                                       retry_on_timeout=retry_on_timeout,
                                                       **kwargs
                                                       )
        else:
            username, password = self._check_auth_details(username, password)
            self.elastic = elasticsearch.Elasticsearch(hosts=hosts,
                                                       basic_auth=(username, password),
                                                       verify_certs=False,
                                                       timeout=timeout,
                                                       max_retries=max_retries,
                                                       retry_on_timeout=retry_on_timeout,
                                                       **kwargs
                                                       )

    def _check_api_auth_details(self, api_username: Optional[str], api_password: Optional[str]):
        if api_username is None:
            api_username = input("API Username: ")
        if api_password is None:
            api_password = getpass.getpass("API Password: ")
        return api_username, api_password

    def _check_auth_details(self, username: Optional[str], password: Optional[str]):
        if username is None:
            username = input("Username:")
        if password is None:
            password = getpass.getpass("Password:")
        return username, password

    def get_docs_generator(self, query: Dict, index: Union[str, List[str]],
                           es_gen_size: int = 800, request_timeout: int = 300, **kwargs):
        """

        :param query: search query body
        :param index: Can be a single index name str or List of ES indices to search.
        :param es_gen_size: Size of the generator object
        :param request_timeout: set to 840000 for large searches
        :return: search generator object
        """
        docs_generator = elasticsearch.helpers.scan(self.elastic,
                                                    query=query,
                                                    index=index,
                                                    size=es_gen_size,
                                                    request_timeout=request_timeout)
        return docs_generator

    def cogstack2df(self, query: Dict, index: Union[str, List[str]], column_headers=None,
                    es_gen_size: int = 800, request_timeout: int = 300):
        """
        Returns DataFrame from a CogStack search

        :param query: search query body
        :param index: str index name or list of indices
        :param column_headers: specify column headers to only retrieve those columns
        :param es_gen_size: Size of the generator to construct df
        :param request_timeout: set to 840000 for large searches
        :return: DataFrame
        """
        docs_generator = elasticsearch.helpers.scan(self.elastic,
                                                    query=query,
                                                    index=index,
                                                    size=es_gen_size,
                                                    request_timeout=request_timeout)
        temp_results = []
        results = self.elastic.count(index=index, query=query['query'])
        for hit in tqdm(docs_generator, total=results['count'], desc="CogStack retrieved... "):
            row = dict()
            row['_index'] = hit['_index']
            row['_type'] = hit['_type']
            row['_id'] = hit['_id']
            row['_score'] = hit['_score']
            row.update(hit['_source'])
            temp_results.append(row)
        if column_headers:
            df_headers = ['_index', '_type', '_id', '_score']
            df_headers.extend(column_headers)
            df = pd.DataFrame(temp_results, columns=df_headers)
        else:
            df = pd.DataFrame(temp_results)
        return df

    def DataFrame(self, index: Optional[str]):
        """
        Special function to return a pandas-like DataFrame that remains in CogStack and not in memory. See cogstack2df func
         to retrieve data to memory.
        :param index: List of indices
        :return: A DataFrame object
        """
        return ed.DataFrame(es_client=self.elastic, es_index_pattern=index)

    # TODO These below functions are legacy and make not work with new ES version
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
            print("Text of the clinical note corrupted: " + text[0:100])


    def bulk_to_cogstack(self):
        # TODO: look the code made for Nazli/Dan
        pass
