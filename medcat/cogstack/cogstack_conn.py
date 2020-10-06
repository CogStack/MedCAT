import os
import json
import datetime
import getpass
import requests
import pandas as pd
from datetime import datetime
from datetime import timedelta
import elasticsearch
import elasticsearch.helpers

from typing import Dict, List

class CogStackConn(object):
    def __init__(self, host, port=9200, username: str=None, password: str=None, scheme: str='https',
                 timeout: int=360, max_retries: int=10, retry_on_timeout: bool=True, **kwargs):
        username, password = self._check_auth_details(username, password)

        self.elastic = elasticsearch.Elasticsearch(hosts=[{'host': host, 'port': port}],
                                         http_auth=(username, password),
                                         scheme=scheme,
                                         verify_certs=False,
                                         timeout=timeout,
                                         max_retries=max_retries,
                                         retry_on_timeout=retry_on_timeout,
                                         **kwargs)

    def _check_auth_details(self, username: str=None, password: str=None):
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


    def bulk_to_cogstack(self):
        # TODO: look the code made for Nazli/Dan
        pass
