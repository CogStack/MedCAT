import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="medcat",
    setup_requires=["setuptools_scm"],
    use_scm_version={"local_scheme": "no-local-version", "fallback_version": "unknown"},
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Concept annotation tool for Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/MedCAT",
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing', 'medcat.cogstack', 'medcat.ner', 'medcat.linking', 'medcat.datasets',
              'medcat.tokenizers', 'medcat.utils.meta_cat', 'medcat.pipeline', 'medcat.neo', 'medcat.utils.ner'],
    install_requires=[
        'numpy>=1.21.4',
        'pandas<=1.4.2,>=1.1.5',
        'gensim~=4.1.2',
        'spacy<3.1.4,>=3.1.0',
        'scipy<=1.8.1,>=1.5.4',
        'transformers~=4.19.2',
        'torch>=1.0',
        'tqdm>=4.27',
        'sklearn~=0.0',
        'elasticsearch>=7.10,<8.0.0',
        'dill~=0.3.4',
        'datasets~=2.2.2',
        'jsonpickle~=2.0.0',
        'psutil<6.0.0,>=5.8.0',
        'multiprocess', # seems to work better than standard mp
        'py2neo==2021.2.3',
        'aiofiles~=0.8.0',
        'ipywidgets~=7.6.5',
        'xxhash==3.0.0',
        'blis<=0.7.5',
        'click<=8.0.4', # Spacy breaks without this
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
