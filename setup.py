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
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing', 'medcat.ner', 'medcat.linking', 'medcat.datasets',
              'medcat.tokenizers', 'medcat.utils.meta_cat', 'medcat.pipeline', 'medcat.neo', 'medcat.utils.ner',
              'medcat.utils.saving', 'medcat.utils.regression'],
    install_requires=[
        'numpy>=1.22.0', # first to support 3.11
        'pandas>=1.4.2', # first to support 3.11
        'gensim>=4.3.0', # first to support 3.11
        'spacy>=3.1.0',
        'scipy~=1.9.2', # first to support 3.11
        'transformers>=4.19.2,<4.22.0', # upper bound is needed for the de-id model until it is retrained
        'tokenizers~=0.12', # 0.13.1 doesn't seem to build
        'torch>=1.13.0', # first to support 3.11
        'tqdm>=4.27',
        'scikit-learn>=1.1.3', # first to supporrt 3.11
        'dill>=0.3.4', # allow later versions with later versions of datasets (tested with 0.3.6)
        'datasets>=2.2.2', # allow later versions, tested with 2.7.1
        'jsonpickle>=2.0.0', # allow later versions, tested with 3.0.0
        'psutil>=5.8.0',
        # 0.70.12 uses older version of dill (i.e less than 0.3.5) which is required for datasets
        'multiprocess~=0.70.12',  # 0.70.14 seemed to work just fine
        'py2neo~=2021.2.3',
        'aiofiles>=0.8.0', # allow later versions, tested with 22.1.0
        'ipywidgets>=7.6.5', # allow later versions, tested with 0.8.0
        'xxhash>=3.0.0', # allow later versions, tested with 3.1.0
        'blis>=0.7.5', # allow later versions, tested with 0.7.9
        'click>=8.0.4', # allow later versions, tested with 8.1.3
        'pydantic>=1.10.0,<2.0', # for spacy compatibility; avoid 2.0 due to breaking changes
        # the following are not direct dependencies of MedCAT but needed for docs/building
        # hopefully will no longer need the transitive dependencies
        'aiohttp==3.8.5', # 3.8.3 is needed for compatibility with fsspec <- datasets <- medcat
        'blis<0.8.0,>=0.7.8', # as required by thinc <- spacy <- medcat
        # 'smart-open==5.2.1', # 5.2.1 is needed for compatibility with pathy
        # 'joblib~=1.2',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
