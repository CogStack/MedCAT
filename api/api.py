from flask import Flask
from cat.umls import UMLS
from cat.utils.spacy_pipe import SpacyPipe
from cat.utils.vocab import Vocab
from cat.cat import CAT
from flask import request

vocab = Vocab()
umls = UMLS()

umls.load_dict('/home/ubuntu/data/umls/models/min-umls-dict-6-2.dat')
vocab.load_dict(path="/home/ubuntu/data/other/dicts/bert-umls-wiki-emb-dict.dat")
cat = CAT(umls, vocab=vocab)
cat.spacy_cat.train = False

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == 'POST':
        return cat.get_json(request.form.get('text'))

    return """
    <form method="POST">
        Text: <input type="text" name="text"><br>
        <input type="submit" value="Submit"><br>
    </form>
    """
