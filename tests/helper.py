import os
import requests
import unittest
import unittest.mock

import numpy as np

from medcat.vocab import Vocab


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
