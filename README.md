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
from cat.preprocessing.vocab import Vocab

vocab = Vocab()
umls = UMLS()

vocab.load_dict('<path to the vocab file>')
umls.load_dict('<path to the umls file>')

cat = CAT(umls=umls, vocab=vocab)

# A simple test
text = "A 14 mm Hemashield tube graft was selected and sewn end-to-end fashion to the proximal aorta using a semi continuous 3-0 Prolene suture."
doc = cat(text)
# All the extracted entites are now in, each one is a spacy Entity:
doc._.ents
# Each entity has an accuracy assigned to it, e.g.
doc._.ents[0]._.acc

# To have a look at the results you can do:
from spacy import displacy
# Note that this will not show all entites, but only the important ones
displacy.serve(doc, style='ent')
```


## Requirements
`python >= 3.5` [tested with 3.7, but most likely works with 3+]

All the rest can be instaled using `pip` from the requirements.txt file, by running:

`pip install -r requirements.txt`
