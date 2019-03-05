from flask import Flask
from cat.umls import UMLS
from cat.utils.spacy_pipe import SpacyPipe
from cat.utils.vocab import Vocab
from cat.cat import CAT
from flask import request
import os

vocab = Vocab()
umls = UMLS()

umls.load_dict('')
vocab.load_dict(path="")
cat = CAT(umls, vocab=vocab)
cat.spacy_cat.train = False

app = Flask(__name__)

@app.route('/api_test', methods=['GET', 'POST'])
def api_test():
    if request.method == 'POST':
        return cat.get_json(request.form.get('text'))

    content = get_file('api_test.html')
    return content


@app.route('/api', methods=['POST'])
def api():
    return cat.get_json(request.get_json(force=True)['text'])


def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)

def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))
