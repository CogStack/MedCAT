# Medical  <img src="https://github.com/w-is-h/cat/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

A simple tool for concept annotation from UMLS or any other source.

### This is still experimental


## How to use
There are a few ways to run CAT

### PIP Installation
`pip install --upgrade medcat`

#### Please install the langauge models before running anything
`python -m spacy download en_core_web_sm`

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz`


### Building a new Concept Database (.csv) or using an existing one
First download the vocabulary from Vocabulary [Download](https://s3-eu-west-1.amazonaws.com/zkcl/med_ann_norm_dict.dat)

Now in python3+ 
```python
from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.prepare_cdb import PrepareCDB
from medcat.cdb import CDB 

vocab = Vocab()

# Load the vocab model you just downloaded
vocab.load_dict('<path to the vocab file>')

# If you have an existing CDB
cdb = CDB()
cdb.load_dict('<path to the cdb file>') 

# If you need a special CDB you can build one from a .csv file
preparator = PrepareCDB(vocab=vocab)
csv_paths = ['<path to your csv_file>', '<another one>', ...] 
# e.g.
csv_paths = ['./examples/simple_cdb.csv']
cdb = preparator.prepare_csvs(csv_paths)

# Save the new CDB for later
cdb.save_dict("<path to a file where it will be saved>")

# To annotate documents we do
doc = "My simple document with kidney failure"
cat = CAT(cdb=cdb, vocab=vocab)
cat.train = False
doc_spacy = cat(doc)
# Entities are in
doc_spacy._.ents
# Or to get a json
doc_json = cat.get_json(doc)

# To have a look at the results:
from spacy import displacy
# Note that this will not show all entites, but only the longest ones
displacy.serve(doc_spacy, style='ent')

# To run cat on a large number of documents
data = [(<doc_id>, <text>), (<doc_id>, <text>), ...]
docs = cat.multi_processing(data)
```

### Training and Fine-tuning
To fine-tune or train everything from the ground up (excluding word-vectors), you can use the following:
```python
# Loadinga CDB or creating a new one is as above.

# To run the training do
f = open("<some file with a lot of medical text>", 'r')
# If you want fine tune set it to True, old training will be preserved
cat.run_training(f, fine_tune=False)
```


## If building from source, the requirements are
`python >= 3.5` [tested with 3.7, but most likely works with 3+]

All the rest can be instaled using `pip` from the requirements.txt file, by running:

`pip install -r requirements.txt`


## Results

| Dataset | SoftF1 | Description |
| --- | :---: | --- |
| MedMentions | 0.83 | The whole MedMentions dataset without any modifications or supervised training |
| MedMentions | 0.828 | MedMentions only for concepts that require disambiguation, or names that map to more CUIs |
| MedMentions | 0.93 | Medmentions filterd by TUI to only concepts that are a disease |


## Models
A basic trained model is made public for the vocabulary. It is trained for the 35K entities available in `MedMentions`. It is quite limited
so the performance might not be the best.

Vocabulary [Download](https://s3-eu-west-1.amazonaws.com/zkcl/med_ann_norm_dict.dat) - Built from MedMentions

(Note: This is was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)


## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The dictionary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words `For now NOT made publicaly available`
