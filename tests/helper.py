import os
import requests
import unittest
import unittest.mock

import numpy as np

from medcat.vocab import Vocab
from medcat.cdb_maker import CDBMaker
from medcat.config import Config


class AsyncMock(unittest.mock.MagicMock):
    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


ERROR_503 = b"""<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>503 Service Unavailable</title>
</head><body>
<h1>Service Unavailable</h1>
<p>The server is temporarily unable to service your
request due to maintenance downtime or capacity
problems. Please try again later.</p>
</body></html>
"""

ERROR_403 = b"""<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<html><head>
<title>403 Forbidden</title>
</head><body>
<h1>Forbidden</h1>
<p>You don't have permission to access this resource.</p>
</body></html>
"""

SIMPLE_WORDS = """house	34444	 0.3232 0.123213 1.231231
dog	14444	0.76762 0.76767 1.45454"""


def generate_simple_vocab():
    v = Vocab()
    # v.add_words()
    for line in SIMPLE_WORDS.split('\n'):
        parts = line.split("\t")
        word = parts[0]
        cnt = int(parts[1].strip())
        vec = None
        if len(parts) == 3:
            vec = np.array([float(x) for x in parts[2].strip().split(" ")])

        v.add_word(word, cnt, vec, replace=True)
    v.make_unigram_table()
    return v


class VocabDownloader:
    url = 'https://cogstack-medcat-example-models.s3.eu-west-2.amazonaws.com/medcat-example-models/vocab.dat'
    vocab_path = "./tmp_vocab.dat"
    _has_simple = False

    def is_valid(self):
        with open(self.vocab_path, 'rb') as f:
            content = f.read()
        if content == ERROR_503:
            return False
        if content == ERROR_403:
            return False
        v = Vocab.load(self.vocab_path)
        if len(v.vocab) == 2:  # simple one
            self._has_simple = True
            return False
        return True

    def check_or_download(self):
        if os.path.exists(self.vocab_path) and self.is_valid():
            return
        tmp = requests.get(self.url)
        if tmp.content == ERROR_503 or tmp.content == ERROR_403:
            print('Rosalind server unavailable')
            if self._has_simple:
                print('Local simple vocab already present')
                return
            print('Generating local simple vocab instead')
            v = generate_simple_vocab()
            v.save(self.vocab_path)
            return
        with open(self.vocab_path, 'wb') as f:
            f.write(tmp.content)


class ForCDBMerging:

    def __init__(self) -> None:
        # generating cdbs - two maker are requested as they point to the same created CDB. 
        config = Config()
        config.general["spacy_model"] = "en_core_web_md"
        maker1 = CDBMaker(config)
        maker2 = CDBMaker(config) # second maker is required as it will otherwise point to same object
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_creator", "umls_sample.csv")
        self.cdb1 = maker1.prepare_csvs(csv_paths=[path])
        self.cdb2 = maker2.prepare_csvs(csv_paths=[path])

        # generating context vectors here for for testing the weighted average function (based off cui2count_train)
        zeroes = np.zeros(shape=(1,300))
        ones = np.ones(shape=(1,300))
        for i, cui in enumerate(self.cdb1.cui2names):
            self.cdb1.cui2context_vectors[cui] = {"short": ones}
            self.cdb2.cui2context_vectors[cui] = {"short": zeroes}
            self.cdb1.cui2count_train[cui] = 1
            self.cdb2.cui2count_train[cui] = i + 1
        # adding new names and cuis to each cdb to test after merging
        test_add = {"test": {'tokens': "test_token", 'snames': ["test_name"], 'raw_name': "test_raw_name", "is_upper": "P"}}
        self.cdb1.add_names("C0006826", test_add)
        unique_test = {"test": {'tokens': "test_token", 'snames': ["test_name"], 'raw_name': "test_raw_name", "is_upper": "P"}}
        self.cdb2.add_names("UniqueTest", unique_test)
        self.cdb2.cui2context_vectors["UniqueTest"] = {"short": zeroes}
        self.cdb2.addl_info["cui2ontologies"] = {}
        self.cdb2.addl_info["cui2description"] = {}
        for cui in self.cdb2.cui2names:
            self.cdb2.addl_info["cui2ontologies"][cui] = {"test_ontology"}
            self.cdb2.addl_info["cui2description"][cui] = "test_description"
