import sys
sys.path.insert(0, "/home/ubuntu/projects/MedCAT/")

from django.shortcuts import render
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.utils.helpers import doc2html
from medcat.vocab import Vocab
from urllib.request import urlretrieve
#from medcat.meta_cat import MetaCAT
from .models import *
import os
import json

vocab_path = os.getenv('VOCAB_PATH', '/tmp/vocab.dat')
cdb_path = os.getenv('CDB_PATH', '/tmp/cdb.dat')

# TODO
#neg_path = os.getenv('NEG_PATH', '/tmp/mc_negated')

try:
    if not os.path.exists(vocab_path):
        vocab_url = os.getenv('VOCAB_URL')
        urlretrieve(vocab_url, vocab_path)

    if not os.path.exists(cdb_path):
        cdb_url = os.getenv('CDB_URL')
        print("*" * 399)
        print(cdb_url)
        urlretrieve(cdb_url, cdb_path)

    vocab = Vocab.load(vocab_path)
    cdb = CDB.load(cdb_path)
    #    mc_negated = MetaCAT(save_dir=neg_path)
    #    mc_negated.load()
    #    cat = CAT(cdb=cdb, vocab=vocab, meta_cats=[mc_negated])
    cat = CAT(cdb=cdb, vocab=vocab, config=cdb.config)
    cat.spacy_cat.MIN_ACC = 0.30
    cat.spacy_cat.MIN_ACC_TH = 0.30
    cat.spacy_cat.ACC_ALWAYS = True
except Exception as e:
    print(str(e))

def get_html_and_json(text):
    doc = cat(text)

    a = json.loads(cat.get_json(text))
    for id, ent in a['annotations'].items():
        new_ent = {}
        for key in ent.keys():
            if key == 'pretty_name':
                new_ent['Pretty Name'] = ent[key]
            if key == 'icd10':
                icd10 = ent.get('icd10', [])
                new_ent['ICD-10 Code'] = icd10[-1]['chapter'] if icd10 else '-'
            new_ent['OPCS Code'] = '-'
            if key == 'opcs':
                opcs = ent.get('opcs', [])
                new_ent['OPCS Code'] = opcs[-1]['chapter'] if opcs else '-'
            if key == 'cui':
                new_ent['Identifier'] = ent[key]
            if key == 'types':
                new_ent['Type'] = ", ".join(ent[key])
            if key == 'acc':
                new_ent['Confidence Score'] = ent[key]
            if key == 'start':
                new_ent['Start Index'] = ent[key]
            if key == 'end':
                new_ent['End Index'] = ent[key]
            if key == 'id':
                new_ent['id'] = ent[key]
            if key == 'meta_anns':
                meta_anns = ent.get("meta_anns", {})
                if meta_anns:
                    for meta_ann in meta_anns.keys():
                        new_ent[meta_ann] = meta_anns[meta_ann]['value']

        a['annotations'][id] = new_ent

    doc_json = json.dumps(a)


    uploaded_text = UploadedText()
    uploaded_text.text = str(text)
    uploaded_text.save()

    return doc2html(doc), doc_json


def train_annotations(request):
    context = {}
    context['doc_json'] = '{"msg": "No documents yet"}'

    if request.POST and 'text' in request.POST:
        doc_html, doc_json = get_html_and_json(request.POST['text'])

        context['doc_html'] = doc_html
        context['doc_json'] = doc_json
        context['text'] = request.POST['text']
    return render(request, 'train_annotations.html', context=context)
