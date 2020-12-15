# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

MedCAT can be used to extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS. Preprint [arXiv](https://arxiv.org/abs/2010.01165). 

## Demo
A demo application is available at [MedCAT](https://medcat.rosalind.kcl.ac.uk). Please note that this was trained on MedMentions
and contains a small portion of UMLS.

## Tutorial
A guide on how to use MedCAT is available in the [tutorial](https://github.com/CogStack/MedCAT/tree/master/tutorial) folder. Read more about MedCAT on [Towards Data Science](https://towardsdatascience.com/medcat-introduction-analyzing-electronic-health-records-e1c420afa13a).

## Papers that use MedCAT
- [Treatment with ACE-inhibitors is not associated with early severe SARS-Covid-19 infection in a multi-site UK acute Hospital Trust](https://www.researchgate.net/publication/340261837_Treatment_with_ACE-inhibitors_is_not_associated_with_early_severe_SARS-Covid-19_infection_in_a_multi-site_UK_acute_Hospital_Trust)
- [Supplementing the National Early Warning Score (NEWS2) for anticipating early deterioration among patients with COVID-19 infection](https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v1)
- [Comparative Analysis of Text Classification Approaches in Electronic Health Records](https://www.researchgate.net/publication/341396173_Comparative_Analysis_of_Text_Classification_Approaches_in_Electronic_Health_Records)
- [Experimental Evaluation and Development of a Silver-Standard for the MIMIC-III Clinical Coding Dataset](https://arxiv.org/abs/2006.07332)

## Related Projects
- [MedCATtrainer](https://github.com/CogStack/MedCATtrainer/) - an interface for building, improving and customising a given Named Entity Recognition and Linking (NER+L) model (MedCAT) for biomedical domain text.
- [MedCATservice](https://github.com/CogStack/MedCATservice) - implements the MedCAT NLP application as a service behind a REST API.
- [iCAT](https://github.com/CogStack/iCAT) - A docker container for CogStack/MedCAT/HuggingFace development in isolated environments.

## Install using PIP (Requires Python 3.6.1+)
1. Install MedCAT 

`pip install --upgrade medcat`

2. Get the scispacy models:

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz`

3. Download the Vocabulary and CDB from the Models section below

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

# Or to get an array of entities, this will return much more information
#and usually easier to use unless you know a lot about spaCy
doc = cat.get_entities(text)
print(doc)
```


## Models
A basic trained model is made public for the vocabulary and CDB. It is trained for the ~ 35K concepts available in `MedMentions`. It is quite limited
so the performance might not be the best.

Vocabulary [Download](https://s3-eu-west-1.amazonaws.com/zkcl/vocab.dat) - Built from MedMentions

CDB [Download](https://s3-eu-west-1.amazonaws.com/zkcl/cdb-medmen.dat) - Built from MedMentions


(Note: This is was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)

### SNOMED-CT and UMLS
If you have access to UMLS or SNOMED-CT and can provide some proof (a screenshot of the [UMLS profile page](https://uts.nlm.nih.gov//uts.html#profile) is perfect, feel free to redact all information you do not want to share), contact us - we are happy to share the pre-built CDB and Vocab for those databases.

Alternatively, you can build the CDBs for scratch from source data. We have used the below steps to build UMLS and SNOMED-CT (UK) for our experiments

#### Building Concept Databases from Scratch
We provide details to build both UMLS and SNOMED-CT concept databases. In both cases CSV files containing the source
data with required columns (column descriptions are provided in the [tutorial](https://colab.research.google.com/drive/1nz2zMDQ3QrlTgpW7FfGaXeV1ZAtZeOe2#scrollTo=ptRmHln9k7hG). 
Given the CSV files the [prepare_cdb.py](https://github.com/CogStack/MedCAT/blob/master/medcat/prepare_cdb.py) script can be used to build a CDB.
 
##### Building a UMLS Concept Database
The UMLS can be downloaded from https://www.nlm.nih.gov/research/umls/index.html in the 
Rich Release Format (RRF). To make subsetting and filtering easier we import UMLS RRF into a PostgreSQL database 
(scripts available at [here](https://github.com/w-is-h/umls)).

Once the data is in the database we can use the following SQL script to download the CSV files containing all concepts 
that will form our CDB.

```
# Selecting concepts for all the Ontologies that are used
SELECT DISTINCT umls.mrconso.cui, str, mrconso.sab, mrconso.tty, tui, sty, def 
FROM umls.mrconso 
    LEFT OUTER JOIN umls.mrsty ON umls.mrsty.cui = umls.mrconso.cui 
    LEFT OUTER JOIN umls.mrdef ON umls.mrconso.cui = umls.mrdef.cui
WHERE lat='ENG'
```

##### Building a SNOMED-CT Concept Database
We use the SNOMED-CT data provided by the NHS TRUD service [https://isd.digital.nhs.uk/trud3/user/guest/group/0/pack/26](https://isd.digital.nhs.uk/trud3/user/guest/group/0/pack/26). 
This release combines the International and UK specific concepts into a set of assets that can be parsed and loaded 
into a MedCAT CDB. We provide scripts for parsing the various release files and load into a MedCAT CDB instance. 
We provide further scripts to load accompanying SNOMED-CT Drug extension and clinical coding data 
(ICD / OPCS terminologies) also from the NHS TRUD service. Scripts are available at: [https://github.com/tomolopolis/SNOMED-CT_Analysis](https://github.com/tomolopolis/SNOMED-CT_Analysis) 


## Acknowledgement
Entity extraction was trained on [MedMentions](https://github.com/chanzuckerberg/MedMentions) In total it has ~ 35K entites from UMLS

The vocabulary was compiled from [Wiktionary](https://en.wiktionary.org/wiki/Wiktionary:Main_Page) In total ~ 800K unique words


## Powered By
A big thank you goes to [spaCy](https://spacy.io/) and [Hugging Face](https://huggingface.co/) - who made life a million times easier.


## Citation
```
@misc{kraljevic2020multidomain,
      title={Multi-domain Clinical Natural Language Processing with MedCAT: the Medical Concept Annotation Toolkit}, 
      author={Zeljko Kraljevic and Thomas Searle and Anthony Shek and Lukasz Roguski and Kawsar Noor and Daniel Bean and Aurelie Mascio and Leilei Zhu and Amos A Folarin and Angus Roberts and Rebecca Bendayan and Mark P Richardson and Robert Stewart and Anoop D Shah and Wai Keong Wong and Zina Ibrahim and James T Teo and Richard JB Dobson},
      year={2020},
      eprint={2010.01165},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
