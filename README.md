# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

[![Build Status](https://github.com/CogStack/MedCAT/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/CogStack/MedCAT/actions/workflows/main.yml?query=branch%3Amaster)
[![Latest release](https://img.shields.io/github/v/release/CogStack/MedCAT)](https://github.com/CogStack/MedCAT/releases/latest)
[![pypi Version](https://img.shields.io/pypi/v/medcat.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/medcat/)

MedCAT can be used to extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS. Paper on [arXiv](https://arxiv.org/abs/2010.01165). 

## News
- **New Minor Release \[20. October 2021\]** Introducing model packs, new faster multiprocessing for large datasets (100M+ documents) and improved MetaCAT. 
- **New Release \[1. August 2021\]**: Upgraded MedCAT to use spaCy v3, new scispaCy models have to be downloaded - all old CDBs (compatble with MedCAT v1) will work without any changes.
- **New Feature and Tutorial \[8. July 2021\]**: [Integrating 🤗 Transformers with MedCAT for biomedical NER+L](https://towardsdatascience.com/integrating-transformers-with-medcat-for-biomedical-ner-l-8869c76762a)
- **General \[1. April 2021\]**: MedCAT is upgraded to v1, unforunately this introduces breaking changes with older models (MedCAT v0.4), 
as well as potential problems with all code that used the MedCAT package. MedCAT v0.4 is available on the legacy 
branch and will still be supported until 1. July 2021 
(with respect to potential bug fixes), after it will still be available but not updated anymore.
- **Paper**: [What’s in a Summary? Laying the Groundwork for Advances in Hospital-Course Summarization](https://www.aclweb.org/anthology/2021.naacl-main.382.pdf)
- ([more...](https://github.com/CogStack/MedCAT/blob/master/media/news.md))

## Demo
A demo application is available at [MedCAT](https://medcat.rosalind.kcl.ac.uk). This was trained on MIMIC-III and all of SNOMED-CT.

## Tutorial
A guide on how to use MedCAT is available in the [tutorial](https://github.com/CogStack/MedCAT/tree/master/tutorial) folder. Read more about MedCAT on [Towards Data Science](https://towardsdatascience.com/medcat-introduction-analyzing-electronic-health-records-e1c420afa13a).

## Related Projects
- [MedCATtrainer](https://github.com/CogStack/MedCATtrainer/) - an interface for building, improving and customising a given Named Entity Recognition and Linking (NER+L) model (MedCAT) for biomedical domain text.
- [MedCATservice](https://github.com/CogStack/MedCATservice) - implements the MedCAT NLP application as a service behind a REST API.
- [iCAT](https://github.com/CogStack/iCAT) - A docker container for CogStack/MedCAT/HuggingFace development in isolated environments.

## Install using PIP (Requires Python 3.6+)
0. Upgrade pip `pip install --upgrade pip`
1. Install MedCAT 
- For macOS/linux: `pip install --upgrade medcat`
- For Windows (see [PyTorch documentation](https://pytorch.org/get-started/previous-versions/)): `pip install --upgrade medcat -f https://download.pytorch.org/whl/torch_stable.html`

2. Quickstart (MedCAT v1.2+):
```python
from medcat.cat import CAT

# Download the model_pack from the models section in the github repo.
cat = CAT.load_model_pack('<path to downloaded zip file>')

# Test it
text = "My simple document with kidney failure"
entities = cat.get_entities(text)
print(entities)

# To run unsupervised training over documents
data_iterator = <your iterator>
cat.train(data_iterator)
#Once done, save the whole model_pack 
cat.create_model_pack(<save path>)
```


3. Quick start with separate models:
New Models (MedCAT v1.2+) need the spacy `en_core_web_md` while older ones use the scispacy models, install the one you need or all if not sure. If using model packs you do not need to download these models: 
```
python -m spacy download en_core_web_md
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_md-0.4.0.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
```
```python
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT
from medcat.meta_cat import MetaCAT

# Load the vocab model you downloaded
vocab = Vocab.load(vocab_path)
# Load the cdb model you downloaded
cdb = CDB.load('<path to the cdb file>') 

# Download the mc_status model from the models section below and unzip it
mc_status = MetaCAT.load("<path to the unziped mc_status directory>")
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=[mc_status])

# Test it
text = "My simple document with kidney failure"
entities = cat.get_entities(text)
print(entities)

# To run unsupervised training over documents
data_iterator = <your iterator>
cat.train(data_iterator)
#Once done you can make the current pipeline into a model_pack 
cat.create_model_pack(<save path>)
```


## Models
A basic trained model is made public. It contains ~ 35K concepts available in `MedMentions`. 

#### ModelPacks

- MedMentions with Status (Is Concept Affirmed or Negated/Hypothetical) [Download](https://medcat.rosalind.kcl.ac.uk/media/medmen_wstatus_2021_oct.zip)


#### Separate models

- Vocabulary [Download](https://medcat.rosalind.kcl.ac.uk/media/vocab.dat) - Built from MedMentions

- CDB [Download](https://medcat.rosalind.kcl.ac.uk/media/cdb-medmen-v1_2.dat) - Built from MedMentions

- MetaCAT Status [Download](https://medcat.rosalind.kcl.ac.uk/media/mc_status.zip) - Built from a sample from MIMIC-III, detects is an annotation Affirmed (Positve) or Other (Negated or Hypothetical)


(Note: This was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)

### SNOMED-CT and UMLS
If you have access to UMLS or SNOMED-CT and can provide some proof (a screenshot of the [UMLS profile page](https://uts.nlm.nih.gov//uts.html#profile) is perfect, feel free to redact all information you do not want to share), contact us - we are happy to share the pre-built CDB and Vocab for those databases. 


## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The vocabulary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words


## Powered By
A big thank you goes to [spaCy](https://spacy.io/) and [Hugging Face](https://huggingface.co/) - who made life a million times easier.


## Citation
```
@ARTICLE{Kraljevic2021-ln,
  title="Multi-domain clinical natural language processing with {MedCAT}: The Medical Concept Annotation Toolkit",
  author="Kraljevic, Zeljko and Searle, Thomas and Shek, Anthony and Roguski, Lukasz and Noor, Kawsar and Bean, Daniel and Mascio, Aurelie and Zhu, Leilei and Folarin, Amos A and Roberts, Angus and Bendayan, Rebecca and Richardson, Mark P and Stewart, Robert and Shah, Anoop D and Wong, Wai Keong and Ibrahim, Zina and Teo, James T and Dobson, Richard J B",
  journal="Artif. Intell. Med.",
  volume=117,
  pages="102083",
  month=jul,
  year=2021,
  issn="0933-3657",
  doi="10.1016/j.artmed.2021.102083"
}
```
