# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

A tool for extraction and linking (UMLS/SNOMED/...) of diseases/drugs/sympotms/... from Electronic Health Reacords or any other free text. Paper on [arXiv](https://arxiv.org/abs/1912.10166). 

## Demo
A demo application is available at [MedCAT](https://medcat.rosalind.kcl.ac.uk). Please note that this was trained on MedMentions
and contains a very small portion of UMLS (<1%). 

## Tutorial
A guide on how to use MedCAT is available in the [tutorial](https://github.com/CogStack/MedCAT/tree/master/tutorial) folder. Read more about MedCAT on [Towards Data Science](https://towardsdatascience.com/medcat-introduction-analyzing-electronic-health-records-e1c420afa13a).

## Papers
[Treatment with ACE-inhibitors is not associated with early severe SARS-Covid-19 infection in a multi-site UK acute Hospital Trust](https://www.researchgate.net/publication/340261837_Treatment_with_ACE-inhibitors_is_not_associated_with_early_severe_SARS-Covid-19_infection_in_a_multi-site_UK_acute_Hospital_Trust)

### Install using PIP
1. Install MedCAT 

`pip install --upgrade medcat`

2. Get the scispacy models:

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz`

3. Downlad the Vocabulary and CDB from the Models section bellow

4. Quickstart:
```python
from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB 

vocab = Vocab()
# Load the vocab model you downloaded
vocab.load_dict('<path to the vocab file>')

# Load the cdb model you downloaded
cdb = CDB()
cdb.load_dict('<path to the cdb file>') 

# create cat
cat = CAT(cdb=cdb, vocab=vocab)

# Test it
text = "My simple document with kidney failure"
doc_spacy = cat(text)
# Print detected entities
print(doc_spacy.ents)

# Or to get a json
import json
doc_json = json.loads(cat.get_json(text))
print(doc_json)
```


## Models
A basic trained model is made public for the vocabulary and CDB. It is trained for the ~ 35K concepts available in `MedMentions`. It is quite limited
so the performance might not be the best.

Vocabulary [Download](https://s3-eu-west-1.amazonaws.com/zkcl/vocab.dat) - Built from MedMentions

CDB [Download](https://s3-eu-west-1.amazonaws.com/zkcl/cdb-medmen.dat) - Built from MedMentions


(Note: This is was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)


## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The vocabulary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words
