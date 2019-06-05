from flask import Flask
from flask import Response
import json
from medcat.cdb import CDB
from medcat.utils.vocab import Vocab
from medcat.cat import CAT
from flask import request
import os


vocab = Vocab()
cdb = CDB()

cdb.load_dict(os.getenv("CDB_MODEL", '/cat/models/med_ann_norm.dat'))
vocab.load_dict(path=os.getenv("VOCAB_MODEL", '/cat/models/med_ann_norm_dict.dat'))
cat = CAT(cdb, vocab=vocab)

cat.spacy_cat.train = False

app = Flask(__name__)


app_name = 'MEDCAT'
app_lang = 'en'
app_version = os.getenv("CAT_VERSION", '0.1.0')


@app.route('/api/info', methods=['GET'])
def info():
    app_info = {'name': app_name,
                'language': app_lang,
                'version': app_version}
    return Response(response=json.dumps(app_info), status=200, mimetype="application/json")


@app.route('/api/process', methods=['POST'])
def process():
    payload = request.get_json()
    if payload is None or not is_request_payload_valid(payload):
        return Response(response="Input Payload should be JSON", status=400)

    try:
        result = process_request_payload(payload)
        response = {'result': result}
        return Response(response=json.dumps(response), status=200)

    except Exception as e:
        Response(response="Internal processing error %d" % e, status=500)


@app.route('/api/process_bulk', methods=['POST'])
def process_bulk():
    payload = request.get_json()
    if payload is None or not is_request_payload_valid_bulk(payload):
        return Response(response="Input Payload should be JSON", status=400)

    try:
        result = process_request_payload_bulk(payload)
        response = {'result': result}
        return Response(response=json.dumps(response), status=200)

    except Exception as e:
        Response(response="Internal processing error %d" % e, status=500)


def is_request_payload_valid(payload):
    if 'content' not in payload or 'text' not in payload['content']:
        return False
    return True


def process_request_payload(payload):
    text = payload['content']['text']
    if text is not None and len(text) > 0:
        entities = cat.get_entities(text)
    else:
        entities = []

    # parse the result
    nlp_result = {'text': text,
                  'annotations': entities,
                  'success': True
                  }

    # append footer
    if 'footer' in payload['content']:
        nlp_result['footer'] = payload['content']['footer']

    return nlp_result


def is_request_payload_valid_bulk(payload):
    if 'content' not in payload:
        return False
    return True


def process_request_payload_bulk(payload):

    # prepare the payload
    content = payload['content']
    documents = []

    for i in range(len(content)):
        # skip blank / empty documents
        if 'text' in content[i] and content[i]['text'] is not None and len(content[i]['text']) > 0:
            documents.append((i, content[i]['text']))

    nproc = int(os.getenv('NPROC', 8))
    batch_size = min(300, int(len(documents) / (2 * nproc)))
    res_raw = cat.multi_processing(documents, nproc=nproc, batch_size=batch_size)

    #assert len(res_raw) == len(payload['content'])
    assert len(res_raw) == len(documents)

    # parse the result -- need to sort by IDs as there's no guarantee
    # on the order of the docs processed
    res_sorted = sorted(res_raw, key=lambda x: x[0])
    nlp_result = []

    # parse the results and add footer if needed
    for i in range(len(res_sorted)):
        res = res_sorted[i]
        in_ct = content[i]

        # parse the result
        out_res = {'text': res[1]["text"],
                   'annotations': res[1]["entities"],
                   'success': True
                   }
        # append the footer
        if 'footer' in in_ct:
            out_res['footer'] = in_ct['footer']

        nlp_result.append(out_res)

    return nlp_result


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
