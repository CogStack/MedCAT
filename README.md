# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

A simple tool for concept annotation from UMLS/SNOMED or any other source. Paper on [arXiv](https://arxiv.org/abs/1912.10166). 

## Demo
A demo application is available at [MedCAT](https://medcat.rosalind.kcl.ac.uk). Please note that this was trained on MedMentions
and contains a very small portion of UMLS (<1%). 


### Install using PIP
1. Install MedCAT 

`pip install --upgrade medcat`

2. Install the scispacy models

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.3/en_core_sci_md-0.2.3.tar.gz`

3. Downlad the Vocabulary and CDB from the Models section bellow

4. How to use:
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
cat.train = False

# Test it
doc = "My simple document with kidney failure"
doc_spacy = cat(doc)
# Entities are in
doc_spacy._.ents
# Or to get a json
doc_json = cat.get_json(doc)

# To have a look at the results:
from spacy import displacy
# Note that this will not show all entites, but only the longest ones
displacy.serve(doc_spacy, style='ent')

# To train - unsupervised, set the train flag to True and run
#documents through MedCAT
cat.train = True

# To run cat on a large number of documents, this will
#also run trainnig as the flag is set to True.
data = [(<doc_id>, <text>), (<doc_id>, <text>), ...]
docs = cat.multi_processing(data)

# To explicitly run trainnig you can do
f = open("<some file with a lot of medical text>", 'r')
# If you want fine tune set it to True, old training will be preserved
cat.run_training(f, fine_tune=True)
```


### Building a new Concept Database

```python
from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB 

vocab = Vocab()
# Load the vocab model you downloaded
vocab.load_dict('<path to the vocab file>')

# If you have an existing CDB
cdb = CDB()
cdb.load_dict('<path to the cdb file>') 

# You can now add concepts from a CSV file, examples of the files can be found in ./examples
preparator = PrepareCDB(vocab=vocab)
csv_paths = ['<path to your csv_file>', '<another one>', ...] 
# e.g.
csv_paths = ['./examples/simple_cdb.csv']
cdb = preparator.prepare_csvs(csv_paths)

# Save the new CDB for later
cdb.save_dict("<path to a file where it will be saved>")
# Done
```

## If building from source, the requirements are
`python >= 3.5`

All the rest can be instaled using `pip` from the requirements.txt file, by running:

`pip install -r requirements.txt`


## Results

| Dataset | SoftF1 | Description |
| --- | :---: | --- |
| MedMentions | 0.84 | The whole MedMentions dataset without any modifications or supervised training |
| MedMentions | 0.828 | MedMentions only for concepts that require disambiguation, or names that map to more CUIs |
| MedMentions | 0.97 | Medmentions filterd by TUI to only concepts that are a disease |


## Models
A basic trained model is made public for the vocabulary and CDB. It is trained for the ~ 35K concepts available in `MedMentions`. It is quite limited
so the performance might not be the best.

Vocabulary [Download](https://s3-eu-west-1.amazonaws.com/zkcl/vocab.dat) - Built from MedMentions

CDB [Download](https://s3-eu-west-1.amazonaws.com/zkcl/cdb-medmen.dat) - Built from MedMentions


(Note: This is was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)


## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The dictionary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words `For now NOT made publicaly available`
