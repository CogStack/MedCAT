import os
import requests
import unittest


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


class VocabDownloader:
    url = 'https://medcat.rosalind.kcl.ac.uk/media/vocab.dat'
    vocab_path = "./tmp_vocab.dat"

    def check_or_download(self):
        if not os.path.exists(self.vocab_path):
            tmp = requests.get(self.url)
            if tmp.content == ERROR_503:
                raise AssertionError("Rosalind server not available!")
            with open(self.vocab_path, 'wb') as f:
                f.write(tmp.content)
        else:
            with open(self.vocab_path, 'rb') as f:
                content = f.read()
            if content == ERROR_503:
                print('ERROR 503 saved as vocab - removing and trying to redownload')
                os.remove(self.vocab_path)
                self.check_or_download()
