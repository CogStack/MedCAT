# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png?raw=true" width=45>oncept Annotation Tool

[![Build Status](https://github.com/CogStack/MedCAT/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/CogStack/MedCAT/actions/workflows/main.yml?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/medcat/badge/?version=latest)](https://medcat.readthedocs.io/en/latest/?badge=latest)
[![Latest release](https://img.shields.io/github/v/release/CogStack/MedCAT)](https://github.com/CogStack/MedCAT/releases/latest)
[![pypi Version](https://img.shields.io/pypi/v/medcat.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/medcat/)

MedCAT can be used to extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS. Paper on [arXiv](https://arxiv.org/abs/2010.01165).

**Official Docs [here](https://medcat.readthedocs.io/en/latest/)**

**Discussion Forum [here](https://discourse.cogstack.org/)**

**Available Models (requires UMLS license) [here](https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback)**

## News
- **Paper** [A New Public Corpus for Clinical Section Identification: MedSecId](https://aclanthology.org/2022.coling-1.326.pdf)
- **New Release** \[5. October 2022\]**: Logging changes, and various small updates. [Full changelog](https://github.com/CogStack/MedCAT/compare/v1.3.0...v1.4.0)
- **New Downloader \[15. March 2022\]**: You can now [download](https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback) the latest SNOMED-CT and UMLS model packs via UMLS user authentication.
- **New Feature and Tutorial \[7. December 2021\]**: [Exploring Electronic Health Records with MedCAT and Neo4j](https://towardsdatascience.com/exploring-electronic-health-records-with-medcat-and-neo4j-f376c03d8eef)
- **New Minor Release \[20. October 2021\]** Introducing model packs, new faster multiprocessing for large datasets (100M+ documents) and improved MetaCAT.
- **New Release \[1. August 2021\]**: Upgraded MedCAT to use spaCy v3, new scispaCy models have to be downloaded - all old CDBs (compatble with MedCAT v1) will work without any changes.
- **New Feature and Tutorial \[8. July 2021\]**: [Integrating 🤗 Transformers with MedCAT for biomedical NER+L](https://towardsdatascience.com/integrating-transformers-with-medcat-for-biomedical-ner-l-8869c76762a)
- **General \[1. April 2021\]**: MedCAT is upgraded to v1, unforunately this introduces breaking changes with older models (MedCAT v0.4),
  as well as potential problems with all code that used the MedCAT package. MedCAT v0.4 is available on the legacy
  branch and will still be supported until 1. July 2021
  (with respect to potential bug fixes), after it will still be available but not updated anymore.

## Demo
A demo application is available at [MedCAT](https://medcat.rosalind.kcl.ac.uk). This was trained on MIMIC-III and all of SNOMED-CT.

## Tutorials
A guide on how to use MedCAT is available at [MedCAT Tutorials](https://github.com/CogStack/MedCATtutorials). Read more about MedCAT on [Towards Data Science](https://towardsdatascience.com/medcat-introduction-analyzing-electronic-health-records-e1c420afa13a).

## Related Projects
- [MedCATtrainer](https://github.com/CogStack/MedCATtrainer/) - an interface for building, improving and customising a given Named Entity Recognition and Linking (NER+L) model (MedCAT) for biomedical domain text.
- [MedCATservice](https://github.com/CogStack/MedCATservice) - implements the MedCAT NLP application as a service behind a REST API.
- [iCAT](https://github.com/CogStack/iCAT) - A docker container for CogStack/MedCAT/HuggingFace development in isolated environments.

## Install using PIP (Requires Python 3.7+)
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

4. Quick start with to create CDB and vocab models using local data and a config file:
```bash
# Run model creator with local config file
python medcat/utils/model_creator.py <path_to_model_creator_config_file>

# Run model creator with example file
python medcat/utils/model_creator.py tests/model_creator/config_example.yml
```

| Model creator parameter | Description |
| -------- | ----------- |
| concept_csv_file | Path to file containing UMLS concepts, including primary names, synonyms, types and source ontology. See [examples](https://github.com/CogStack/MedCAT/tree/master/examples) and [tests/model_creator/umls_sample.csv](https://github.com/CogStack/MedCAT/tree/master/tests/model_creator/umls_sample.csv) for format description and examples. |
| unsupervised_training_data_file | Path to file containing text dataset used for spell checking and unsupervised training.|
| output_dir | Path to output directory for writing the CDB and vocab models. |
| medcat_config_file | Path to optional config file for adjusting MedCAT properties, see [configs](https://github.com/CogStack/MedCAT/tree/master/configs), [medcat/config.py](https://github.com/CogStack/MedCAT/tree/master/medcat/config.py) and [tests/model_creator/medcat.txt](https://github.com/CogStack/MedCAT/tree/master/tests/model_creator/medcat.txt)| 
| unigram_table_size | Optional parameter for setting the initialization size of the unigram table in the vocab model. Default is 100000000, while for testing with a small unsupervised training data file a much smaller size could work. | 

## Models
### SNOMED-CT and UMLS
If you have access to UMLS or SNOMED-CT, you can download the pre-built CDB and Vocab for those databases by signing in and filling out [the online form](https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback). This link first requires you to authenticate your ontology access via the NIH portal.

### MedMentions
A basic trained model is made public. It contains ~ 35K concepts available in `MedMentions`. This was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as that data is not publicaly available.

Model packs:
- MedMentions with Status (Is Concept Affirmed or Negated/Hypothetical) [Download](https://medcat.rosalind.kcl.ac.uk/media/medmen_wstatus_2021_oct.zip)

Separate models:
- Vocabulary [Download](https://medcat.rosalind.kcl.ac.uk/media/vocab.dat) - Built from MedMentions
- CDB [Download](https://medcat.rosalind.kcl.ac.uk/media/cdb-medmen-v1_2.dat) - Built from MedMentions
- MetaCAT Status [Download](https://medcat.rosalind.kcl.ac.uk/media/mc_status.zip) - Built from a sample from MIMIC-III, detects is an annotation Affirmed (Positve) or Other (Negated or Hypothetical)

## Acknowledgements
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
