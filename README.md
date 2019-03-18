# <img src="https://github.com/w-is-h/cat/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

A simple tool for concept annotation from UMLS or any other source.

### This is still experimental


## How to use
There are a few ways to run CAT, simplest one being docker.

### Docker
If using docker the appropriate models will be automatically downloaded, you only need to run:

`docker build --network=host -t cat -f Dockerfile.MedMen .`

Once the container is built start it using:

`docker run --env-file=./envs/env_medann -p 5000:5000 cat`

You can now access the API on

`<YOUR_IP>:5000/api`

OR a simple frontend

`<YOUR_IP>:5000/api_test`


### From IPython/Jupyter/Python

First export the path to the `cat` repo:

`export PYTHONPATH=/home/user/cat/`


Given that you already have a Vocab and the UMLS class prebuilt you only need to do the following:
```python
from cat.cat import CAT
from cat.umls import UMLS
from cat.utils.vocab import Vocab

vocab = Vocab()
umls = UMLS()

# Models available for download, look bellow
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
`python -m spacy download en_core_web_sm`

All the rest can be instaled using `pip` from the requirements.txt file, by running:

`pip install -r requirements.txt`


## Results

| Dataset | SoftF1 | Description |
| --- | :---: | --- |
| MedMentions | 0.798 | The whole MedMentions dataset without any modifications or supervised training |
| MedMentions | 0.786 | MedMentions only for concepts that require disambiguation, or names that map to more CUIs |
| MedMentions | 0.92 | Medmentions filterd by TUI to only concepts that are a disease |


## Models
A basic trained model is made public. It is trained for the 35K entities available in `MedMentions`. It is quite limited
so the performance might not be the best.

Vocabulary [Download](https://s3-eu-west-1.amazonaws.com/zkcl/med_ann_norm_dict.dat) - Built from MedMentions

Trained UMLS [Download](https://s3-eu-west-1.amazonaws.com/zkcl/med_ann_norm.dat)

(Note: This is was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)


## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The dictionary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words `For now NOT made publicaly available`
