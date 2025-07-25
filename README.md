# ⚠️ This Project Has Moved Repository


<div align="center">

[![Project Moved](https://img.shields.io/badge/🚀-Project%20Moved-brightgreen)](https://github.com/CogStack/cogstack-nlp)

👉 Please visit the new repository here:  
**[https://github.com/CogStack/cogstack-nlp](https://github.com/CogStack/cogstack-nlp)**

[<img src="https://avatars.githubusercontent.com/u/28688163" width="45" />](https://github.com/CogStack/cogstack-nlp)

</div>

---

<br>

# Medical  <img src="https://raw.githubusercontent.com/CogStack/MedCAT/main/media/cat-logo.png" width=45> oncept Annotation Tool

**NB! [MedCAT v2](https://github.com/CogStack/MedCAT2) will soon be released.**
**MedCAT v1.16.0 will be the latest MedCAT v1 minor release.**
**However, we will likely continue to support v1.16.x with patch releases for some time.**

[![Build Status](https://github.com/CogStack/MedCAT/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/CogStack/MedCAT/actions/workflows/main.yml?query=branch%3Amain)
[![Documentation Status](https://readthedocs.org/projects/medcat/badge/?version=latest)](https://medcat.readthedocs.io/en/latest/?badge=latest)
[![Latest release](https://img.shields.io/github/v/release/CogStack/MedCAT)](https://github.com/CogStack/MedCAT/releases/latest)
[![pypi Version](https://img.shields.io/pypi/v/medcat.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/medcat/)

MedCAT can be used to extract information from Electronic Health Records (EHRs) and link it to biomedical ontologies like SNOMED-CT and UMLS. Paper on [arXiv](https://arxiv.org/abs/2010.01165). 

**Official Docs [here](https://medcat.readthedocs.io/en/latest/)**

**Discussion Forum [discourse](https://discourse.cogstack.org/)**

## Available Models

We have 4 public models available:
1) UMLS Small (A modelpack containing a subset of UMLS (disorders, symptoms, medications...). Trained on MIMIC-III)
2) SNOMED International (Full SNOMED modelpack trained on MIMIC-III)
3) UMLS Dutch v1.10 (a modelpack provided by UMC Utrecht containing [UMLS entities with Dutch names](https://github.com/umcu/dutch-umls) trained on Dutch medical wikipedia articles and a negation detection model [repository](https://github.com/umcu/negation-detection/)/[paper](https://doi.org/10.48550/arxiv.2209.00470) trained on EMC Dutch Clinical Corpus).
4) UMLS Full. >4MM concepts trained self-supervised on MIMIC-III. v2022AA of UMLS.

To download any of these models, please [follow this link](https://uts.nlm.nih.gov/uts/login?service=https://medcat.sites.er.kcl.ac.uk/auth-callback) (or [this link for API key based download](https://medcat.sites.er.kcl.ac.uk/auth-callback-api)) and sign into your NIH profile / UMLS license. You will then be redirected to the MedCAT model download form. Please complete this form and you will be provided a download link.

## News
- **Paper** van Es, B., Reteig, L.C., Tan, S.C. et al. [Negation detection in Dutch clinical texts: an evaluation of rule-based and machine learning methods](https://doi.org/10.1186/s12859-022-05130-x). BMC Bioinformatics 24, 10 (2023). 
- **New tool in the Cogstack ecosystem \[19. December 2022\]** [Foresight -- Deep Generative Modelling of Patient Timelines using Electronic Health Records](https://arxiv.org/abs/2212.08072)
- **New Paper using MedCAT \[21. October 2022\]**: [A New Public Corpus for Clinical Section Identification: MedSecId.](https://aclanthology.org/2022.coling-1.326.pdf)
- **Major Change to the Permissions of Use \[4. August 2022\]** MedCAT now uses the [Elastic License 2.0](https://github.com/CogStack/MedCAT/pull/271/commits/c9f4e86116ec751a97c618c97dadaa23e1feb6bc). For further information please click [here.](https://www.elastic.co/licensing/elastic-license)
- **New Downloader \[15. March 2022\]**: You can now [download](https://uts.nlm.nih.gov/uts/login?service=https://medcat.sites.er.kcl.ac.uk/auth-callback) (or [API key based download](https://medcat.sites.er.kcl.ac.uk/auth-callback-api)) the latest SNOMED-CT and UMLS model packs via UMLS user authentication.
- **New Feature and Tutorial \[7. December 2021\]**: [Exploring Electronic Health Records with MedCAT and Neo4j](https://towardsdatascience.com/exploring-electronic-health-records-with-medcat-and-neo4j-f376c03d8eef)
- **New Minor Release \[20. October 2021\]** Introducing model packs, new faster multiprocessing for large datasets (100M+ documents) and improved MetaCAT.
- **New Release \[1. August 2021\]**: Upgraded MedCAT to use spaCy v3, new scispaCy models have to be downloaded - all old CDBs (compatble with MedCAT v1) will work without any changes.
- **New Feature and Tutorial \[8. July 2021\]**: [Integrating 🤗 Transformers with MedCAT for biomedical NER+L](https://towardsdatascience.com/integrating-transformers-with-medcat-for-biomedical-ner-l-8869c76762a)
- **General \[1. April 2021\]**: MedCAT is upgraded to v1, unforunately this introduces breaking changes with older models (MedCAT v0.4),
  as well as potential problems with all code that used the MedCAT package. MedCAT v0.4 is available on the legacy
  branch and will still be supported until 1. July 2021
  (with respect to potential bug fixes), after it will still be available but not updated anymore.
- **Paper**: [What’s in a Summary? Laying the Groundwork for Advances in Hospital-Course Summarization](https://www.aclweb.org/anthology/2021.naacl-main.382.pdf)
- ([more...](https://github.com/CogStack/MedCAT/blob/main/media/news.md))

## Installation
To install the latest version of MedCAT run the following command:
```
pip install medcat
```
Normal installations of MedCAT will install torch-gpu and all relevant dependancies (such as CUDA). This can require as much as 10 GB more disk space, which isn't required for CPU only usage.

To install the latest version of MedCAT without torch GPU support run the following command:
```
pip install medcat --extra-index-url https://download.pytorch.org/whl/cpu/
```
## Demo
A demo application is available at [MedCAT](https://medcat.sites.er.kcl.ac.uk). This was trained on MIMIC-III and all of SNOMED-CT.
PS: This link can take a long time to load the first time around. The machine spins up as needed and spins down when inactive.

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
