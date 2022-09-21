# Medical  <img src="https://github.com/CogStack/MedCAT/blob/master/media/cat-logo.png" width=45> oncept Annotation Tool

[![Build Status](https://github.com/CogStack/MedCAT/actions/workflows/main.yml/badge.svg?branch=master)](https://github.com/CogStack/MedCAT/actions/workflows/main.yml?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/medcat/badge/?version=latest)](https://medcat.readthedocs.io/en/latest/?badge=latest)
[![Latest release](https://img.shields.io/github/v/release/CogStack/MedCAT)](https://github.com/CogStack/MedCAT/releases/latest)
[![pypi Version](https://img.shields.io/pypi/v/medcat.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/medcat/)

MedCAT can be used to extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS. Paper on [arXiv](https://arxiv.org/abs/2010.01165). 

**Official Docs [here](https://medcat.readthedocs.io/en/latest/)**

**Discussion Forum [discourse](https://discourse.cogstack.org/)**

## Available Models

We have 3 public models available:
1) UMLS Small (A modelpack containing a subset of UMLS (disorders, symptoms, medications...). Trained on MIMIC-III)
2) SNOMED International (Full SNOMED modelpack trained on MIMIC-III)
3) UMLS Dutch v 1.10 (UMLS Dutch (A Dutch UMLS model pack trained on Dutch medical Wikipedia articles and EMC Dutch Clinical Corpus. Provided by Sander Tan, UMC Utrecht, more info at https://github.com/umcu/dutch-umls))

To download any of these models, please [follow this link](https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback) and sign into your NIH profile / UMLS license. You will then be redirected to the MedCAT model download form. Please complete this form and you will be provided a download link.

## News
- **New Downloader \[15. March 2022\]**: You can now [download](https://uts.nlm.nih.gov/uts/login?service=https://medcat.rosalind.kcl.ac.uk/auth-callback) the latest SNOMED-CT and UMLS model packs via UMLS user authentication.
- **New Feature and Tutorial \[7. December 2021\]**: [Exploring Electronic Health Records with MedCAT and Neo4j](https://towardsdatascience.com/exploring-electronic-health-records-with-medcat-and-neo4j-f376c03d8eef)
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

## Tutorials
A guide on how to use MedCAT is available at [MedCAT Tutorials](https://github.com/CogStack/MedCATtutorials). Read more about MedCAT on [Towards Data Science](https://towardsdatascience.com/medcat-introduction-analyzing-electronic-health-records-e1c420afa13a).

## Logging
Since MedCAT is primarily a library, logging has been effectively disabled by default. The idea is that the user of the library should have the choice of what, where, and how to log the information from a specific library they are using.

The idea is that the user can directly modify the logging behaviour of either the entire library or a certain set of modules within as they wish. We have provided a convenience method to add default handlers that log into the console as well as _medcat.log_ (`medcat.add_default_log_handlers`).

Some details as to how one can configure the logging are described in the [MedCAT Tutorials](https://github.com/CogStack/MedCATtutorials).

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
