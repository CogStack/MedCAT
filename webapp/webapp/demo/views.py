import sys
sys.path.insert(0, '/home/ubuntu/projects/MedCAT/')
import os
import json
from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
from wsgiref.util import FileWrapper
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.utils.helpers import doc2html
from medcat.vocab import Vocab
from urllib.request import urlretrieve, urlopen
from urllib.error import HTTPError
#from medcat.meta_cat import MetaCAT
from .models import *
from .forms import DownloaderForm

vocab_path = os.getenv('VOCAB_PATH', '/tmp/vocab.dat')
cdb_path = os.getenv('CDB_PATH', '/tmp/cdb.dat')
AUTH_CALLBACK_SERVICE = 'https://medcat.rosalind.kcl.ac.uk/auth_callback'
VALIDATION_BASE_URL = 'https://uts-ws.nlm.nih.gov/rest/isValidServiceValidate'
VALIDATION_LOGIN_URL = f'https://uts.nlm.nih.gov/uts/login?service={AUTH_CALLBACK_SERVICE}'

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


def validate_umls_user(request):
    ticket = request.GET.get('ticket', '')
    validate_url = f'{VALIDATION_BASE_URL}?service={AUTH_CALLBACK_SERVICE}&ticket={ticket}'
    try:
        is_valid = urlopen(validate_url, timeout=10).read().decode('utf-8')
        context = {
            'is_valid': is_valid == 'true'
        }
        if is_valid == 'true':
            context["message"] = "License verified! Please fill in the following form before downloading models."
            context["downloader_form"] = DownloaderForm(MedcatModel.objects.all())
        else:
            context["message"] = "License not found. Please request or renew your UMLS Metathesaurus License."
    except HTTPError:
        context = {
            'is_valid': False,
            'message': 'Something went wrong. Please try again.'
        }
    finally:
        return render(request, 'umls_user_validation.html', context=context)


def download_model(request):
    if request.method == 'POST':
        downloader_form = DownloaderForm(MedcatModel.objects.all(), request.POST)
        if downloader_form.is_valid():
            downloader_form.save()
            mp_name = downloader_form.cleaned_data['modelpack']
            model = MedcatModel.objects.get(model_name=mp_name)
            if model is not None:
                mp_path = model.model_file.path
            else:
                return HttpResponse(f'Error: Unknown model "{downloader_form.modelpack}"')
            resp = StreamingHttpResponse(FileWrapper(open(mp_path, 'rb')))
            resp["Content-Type"] = "application/zip"
            resp["Content-Length"] = os.path.getsize(mp_path)
            resp["Content-Disposition"] = f"attachment; filename={os.path.basename(mp_path)}"
            return resp
        else:
            return HttpResponse(f'Erorr: All non-optional fields must be filled out. Please <a href="{VALIDATION_LOGIN_URL}">try again</a>.')
    else:
        return HttpResponse('Erorr: Unknown HTTP method.')
