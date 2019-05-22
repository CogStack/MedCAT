from flask import Flask
from medcat.cdb import CDB
from medcat.utils.spacy_pipe import SpacyPipe
from medcat.utils.vocab import Vocab
from medcat.cat import CAT
from flask import request
import os
import json
from spacy import displacy

vocab = Vocab()
cdb = CDB()
cdb.load_dict(os.getenv("CDB_MODEL", '/cat/models/med_ann_norm.dat'))
vocab.load_dict(path=os.getenv("VOCAB_MODEL", '/cat/models/med_ann_norm_dict.dat'))
cat = CAT(cdb, vocab=vocab)
cat.spacy_cat.train = False

app = Flask(__name__)

@app.route('/api_test', methods=['GET', 'POST'])
def api_test():
    if request.method == 'POST':
        return cat.get_json(request.form.get('text'))

    content = get_file('api_test.html')
    return content

@app.route('/doc', methods=['POST'])
def show_annotated_document():
    doc = cat(request.form.get('text'))
    return displacy.render(doc, style='ent')


@app.route('/api', methods=['POST'])
def api():
    return cat.get_json(request.get_json(force=True)['text'])

@app.route('/api_bulk', methods=['POST'])
def api_bulk():
    # Documents must have the following format:
    #[(id, text), (id, text), ...]
    docs = request.get_json(force=True)['documents']
    nproc = int(os.getenv('NPROC', 8))
    batch_size = min(300, int(len(docs) / (2*nproc)))
    res = cat.multi_processing(docs, nproc=nproc, batch_size=batch_size)
    res = json.dumps(res)
    return res

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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
