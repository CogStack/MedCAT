# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

MedCAT can be used to extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS. Paper on [arXiv](https://arxiv.org/abs/2010.01165). 

## UPDATE
MedCAT is upgraded to v1, unforunately this introduces breaking changes with older models (MedCAT v0.4), as well as potential problems with all code that used the MedCAT package.

MedCAT v0.4 is available on the [legacy](https://github.com/CogStack/MedCAT/tree/legacy) branch and will still be supported until 1. July 2021 (with respect to potential bug fixes), after it will still be available but not updated anymore.

## Demo
A demo application is available at [MedCAT](https://medcat.rosalind.kcl.ac.uk). This was trained on MIMIC-III and all of SNOMED-CT.

## Tutorial
A guide on how to use MedCAT is available in the [tutorial](https://github.com/CogStack/MedCAT/tree/master/tutorial) folder. Read more about MedCAT on [Towards Data Science](https://towardsdatascience.com/medcat-introduction-analyzing-electronic-health-records-e1c420afa13a).

## Papers that use MedCAT
- [Treatment with ACE-inhibitors is not associated with early severe SARS-Covid-19 infection in a multi-site UK acute Hospital Trust](https://www.researchgate.net/publication/340261837_Treatment_with_ACE-inhibitors_is_not_associated_with_early_severe_SARS-Covid-19_infection_in_a_multi-site_UK_acute_Hospital_Trust)
- [Supplementing the National Early Warning Score (NEWS2) for anticipating early deterioration among patients with COVID-19 infection](https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v1)
- [Comparative Analysis of Text Classification Approaches in Electronic Health Records](https://www.researchgate.net/publication/341396173_Comparative_Analysis_of_Text_Classification_Approaches_in_Electronic_Health_Records)
- [Experimental Evaluation and Development of a Silver-Standard for the MIMIC-III Clinical Coding Dataset](https://arxiv.org/abs/2006.07332)
- [What’s in a Summary? Laying the Groundwork for Advances in Hospital-Course Summarization](https://www.aclweb.org/anthology/2021.naacl-main.382.pdf)

## Related Projects
- [MedCATtrainer](https://github.com/CogStack/MedCATtrainer/) - an interface for building, improving and customising a given Named Entity Recognition and Linking (NER+L) model (MedCAT) for biomedical domain text.
- [MedCATservice](https://github.com/CogStack/MedCATservice) - implements the MedCAT NLP application as a service behind a REST API.
- [iCAT](https://github.com/CogStack/iCAT) - A docker container for CogStack/MedCAT/HuggingFace development in isolated environments.

## Install using PIP (Requires Python 3.6.1+)
0. Upgrade pip `pip install --upgrade pip`
1. Install MedCAT 
- For macOS/linux: `pip install --upgrade medcat`
- For Windows (see [PyTorch documentation](https://pytorch.org/get-started/previous-versions/)): `pip install --upgrade medcat -f https://download.pytorch.org/whl/torch_stable.html`

2. Get the scispacy models:

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_md-0.3.0.tar.gz`

`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_lg-0.3.0.tar.gz`

3. Downlad the Vocabulary and CDB from the Models section bellow

4. Quickstart:
```python
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT

# Load the vocab model you downloaded
vocab = Vocab.load(vocab_path)
# Load the cdb model you downloaded
cdb = CDB.load('<path to the cdb file>') 

# Create cat - each cdb comes with a config that was used
#to train it. You can change that config in any way you want, before or after creating cat.
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)

# Test it
text = "My simple document with kidney failure"
doc_spacy = cat(text)
# Print detected entities
print(doc_spacy.ents)

# Or to get an array of entities, this will return much more information
#and usually easier to use unless you know a lot about spaCy
doc = cat.get_entities(text)
print(doc)


# To train on one example
_ = cat(text, do_train=True)

# To train on a iterator over documents
data_iterator = <your iterator>
cat.train(data_iterator)

#Once done, save the new CDB
cat.cdb.save(<save path>)
```

### MetaCAT example
```python
from medcat.meta_cat import MetaCAT
# Assume we have a CDB and Vocab object from before
# Download the mc_status model from the models section below and unzip it

mc_status = MetaCAT.load("<path to the unziped mc_status directory>")
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab, meta_cats=[mc_status])

# Now annotate a document, it will have the meta annotation 'status'
doc = cat.get_entities(text)
```


## Models
A basic trained model is made public for the vocabulary and CDB. It is trained for the ~ 35K concepts available in `MedMentions`. 

Vocabulary [Download](https://medcat.rosalind.kcl.ac.uk/media/vocab.dat) - Built from MedMentions

CDB [Download](https://medcat.rosalind.kcl.ac.uk/media/cdb-medmen-v1.dat) - Built from MedMentions

MetaCAT Status [Download](https://medcat.rosalind.kcl.ac.uk/media/mc_status.zip) - Built from a sample from MIMIC-III, detects is an annotation Affirmed (Positve) or Other (Negated or Hypothetical)


(Note: This is was compiled from MedMentions and does not have any data from [NLM](https://www.nlm.nih.gov/research/umls/) as
that data is not publicaly available.)

### SNOMED-CT and UMLS
If you have access to UMLS or SNOMED-CT and can provide some proof (a screenshot of the [UMLS profile page](https://uts.nlm.nih.gov//uts.html#profile) is perfect, feel free to redact all information you do not want to share), contact us - we are happy to share the pre-built CDB and Vocab for those databases. 


## TODO
- [ ] Switch to spaCy version 3+
- [ ] Enable automatic download of pre-built UMLS/SNOMED databases
- [ ] Enable spaCy serialization of documents (problem with `doc._.ents`)
- [ ] Update webapp to v1 and enable UMLS and SNOMED
- [ ] Fix logging, make sure the config options are respected 
- [ ] Relation extraction
- [ ] Implement replace_center in the __call__ function for meta_cat
- [ ] Fix parallelization for MedCAT alone + Try to solve how to run this when we have MetaCATs also
- [ ] How to continue training after unsupervised training (without reseting annealing) - how not to overfit if we have very little annotations, but also not underfit (not learn anything)


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
