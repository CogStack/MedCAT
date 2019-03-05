# Concept Annotation Tool

A simple tool for concept annotation from UMLS or any other source, similar to BioYODIE or SemEHR.

### This is still experimental, it may not work all the time and code is not nicely organized and difficult to read.

## How to use
First export the path to the `cat` repo:

`export PYTHONPATH=/home/user/cat/`


Given that you already have a Vocab and the UMLS class prebuilt you only need to do the following:
```
from cat.cat import CAT
from cat.umls import UMLS
from cat.utils.vocab import Vocab

vocab = Vocab()
umls = UMLS()

vocab.load_dict('<path to the vocab file>') # Use the existing vocab in models
umls.load_dict('<path to the umls file>') # Pretrained models available in models

cat = CAT(umls=umls, vocab=vocab)
# Disable the training
cat.spacy_cat.train = False

# A simple test
text = "A 14 mm Hemashield tube graft was selected and sewn end-to-end fashion to the proximal aorta using a semi continuous 3-0 Prolene suture."
doc = cat(text)
# All the extracted entites are now in, each one is a spacy Entity:
doc._.ents
# Each entity has an accuracy assigned to it, e.g.
doc._.ents[0]._.acc

# To have a look at the results:
from spacy import displacy
# Note that this will not show all entites, but only the longest ones
displacy.serve(doc, style='ent')
```


## Requirements
`python >= 3.5` [tested with 3.7, but most likely works with 3+]

Get the spacy `en` model
`python -m spacy download en`

All the rest can be instaled using `pip` from the requirements.txt file, by running:

`pip install -r requirements.txt`


## Results
Preliminary results show an accuracy of 78% on the MedMentions dataset.



## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The dictionary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words
