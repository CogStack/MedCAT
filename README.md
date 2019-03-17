# Concept Annotation Tool

A simple tool for concept annotation from UMLS or any other source, similar to BioYODIE or SemEHR.

### This is still experimental, it may not work all the time and code is not nicely organized and difficult to read.

## How to use
First export the path to the `cat` repo:

`export PYTHONPATH=/home/user/cat/`


Given that you already have a Vocab and the UMLS class prebuilt you only need to do the following:
```python
from cat.cat import CAT
from cat.umls import UMLS
from cat.utils.vocab import Vocab

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

# To have a look at the results:
from spacy import displacy
# Note that this will not show all entites, but only the longest ones
displacy.serve(doc, style='ent')

# To get JSON output do
doc_json = cat.get_json(text)

# To run cat on a large number of documents
data = [(<doc_id>, <text>), (<doc_id>, <text>), ...]
docs = cat.multi_processing(data)
```

### Training and Fine-tuning

To fine-tune or train everything from the ground up (excluding word-vectors), you can use the following:
```python
from cat.cat import CAT
from cat.umls import UMLS
from cat.utils.vocab import Vocab

vocab = Vocab()
umls = UMLS()

vocab.load_dict('<path to the vocab file>')
umls.load_dict('<path to the umls file>')
cat = CAT(umls=umls, vocab=vocab)

# To run the training do
f = open("<some file with a lot of medical text>", 'r')
# If you want fine tune set it to True, old training will be preserved
cat.run_training(f, fine_tune=False)
```


## Requirements
`python >= 3.5` [tested with 3.7, but most likely works with 3+]

Get the spacy `en` model
`python -m spacy download en`

All the rest can be instaled using `pip` from the requirements.txt file, by running:

`pip install -r requirements.txt`


## Results
Preliminary results show an accuracy of 78% on the MedMentions dataset.

| Dataset | Soft-F1 | Description |
| --- | :---: | --- |
| MedMentions | 0.798 | The whole MedMentions dataset without any modifications or supervised training |
| MedMentions | 0.786 | MedMentions only for concepts that require disambiguation, or names that map to more CUIs |
| MedMentions | 0.92 | Medmentions filterd by TUI to only concepts that are a disease |


## Models
A basic trained model is made public. It is trained for the 35K entities available in `MedMentions`. It is quite limited
so the performance might not be the best.

Vocabulary [Download](https://drive.google.com/file/d/1OJ6UTcm6JrJBN8Rx0Ykjg1uWuS17DPr3/view?usp=sharing)

Trained UMLS [Download](https://drive.google.com/file/d/1KPUdFFTTiD8Wp2xHr9QX-tTwnZ143twd/view?usp=sharing)



## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The dictionary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words `For now NOT made publicaly available`
