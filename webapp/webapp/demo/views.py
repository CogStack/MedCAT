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

AUTH_CALLBACK_SERVICE = 'https://medcat.rosalind.kcl.ac.uk/auth-callback'
VALIDATION_BASE_URL = 'https://uts-ws.nlm.nih.gov/rest/isValidServiceValidate'
VALIDATION_LOGIN_URL = f'https://uts.nlm.nih.gov/uts/login?service={AUTH_CALLBACK_SERVICE}'

model_pack_path = os.getenv('MODEL_PACK_PATH', 'models/medmen_wstatus_2021_oct.zip')

try:
    cat = CAT.load_model_pack(model_pack_path)
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
                new_ent['ICD-10 Code'] = icd10[-1] if icd10 else '-'
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
    uploaded_text.text = len(str(text))#str(text) no saving of text anymore
    uploaded_text.save()

    return doc2html(doc), doc_json


def show_annotations(request):
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
            context['message'] = 'License verified! Please fill in the following form before downloading models.'
            context['downloader_form'] = DownloaderForm(MedcatModel.objects.all())
        else:
            context['message'] = f'License not found. Please request or renew your UMLS Metathesaurus License. If you think you have got the license, try {VALIDATION_LOGIN_URL} again.'
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
            mp_name = downloader_form.cleaned_data['modelpack']
            model = MedcatModel.objects.get(model_name=mp_name)
            if model is not None:
                mp_path = model.model_file.path
            else:
                return HttpResponse(f'Error: Unknown model "{downloader_form.modelpack}"')
            resp = StreamingHttpResponse(FileWrapper(open(mp_path, 'rb')))
            resp['Content-Type'] = 'application/zip'
            resp['Content-Length'] = os.path.getsize(mp_path)
            resp['Content-Disposition'] = f'attachment; filename={os.path.basename(mp_path)}'
            downloader_form.instance.downloaded_file = os.path.basename(mp_path)
            downloader_form.save()
            return resp
        else:
            context = {
                'is_valid': True,
                'downloader_form': downloader_form,
                'message': 'All non-optional fields must be filled out:'
            }
            return render(request, 'umls_user_validation.html', context=context)
    else:
        return HttpResponse('Erorr: Unknown HTTP method.')
