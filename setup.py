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
              'medcat.tokenizers', 'medcat.utils.meta_cat', 'medcat.pipeline', 'medcat.utils.ner',
              'medcat.utils.saving', 'medcat.utils.regression', 'medcat.stats'],
    install_requires=[
        'numpy>=1.22.0,<1.26.0',  # 1.22.0 is first to support python 3.11; post 1.26.0 there's issues with scipy
        'pandas>=1.4.2', # first to support 3.11
        'gensim>=4.3.0,<5.0.0',  # 5.3.0 is first to support 3.11; avoid major version bump
        'spacy>=3.6.0,<4.0.0',  # Some later model packs (e.g HPO) are made with 3.6.0 spacy model; avoid major version bump
        'scipy~=1.9.2',  # 1.9.2 is first to support 3.11
        'transformers>=4.34.0,<5.0.0',  # avoid major version bump
        'accelerate>=0.23.0', # required by Trainer class in de-id
        'torch>=1.13.0,<3.0.0', # 1.13 is first to support 3.11; 2.1.2 has been compatible, but avoid major 3.0.0 for now
        'tqdm>=4.27',
        'scikit-learn>=1.1.3,<2.0.0',  # 1.1.3 is first to supporrt 3.11; avoid major version bump
        'dill>=0.3.6,<1.0.0', # stuff saved in 0.3.6/0.3.7 is not always compatible with 0.3.4/0.3.5; avoid major bump
        'datasets>=2.2.2,<3.0.0', # avoid major bump
        'jsonpickle>=2.0.0', # allow later versions, tested with 3.0.0
        'psutil>=5.8.0',
        # 0.70.12 uses older version of dill (i.e less than 0.3.5) which is required for datasets
        'multiprocess~=0.70.12',  # 0.70.14 seemed to work just fine
        'aiofiles>=0.8.0', # allow later versions, tested with 22.1.0
        'ipywidgets>=7.6.5', # allow later versions, tested with 0.8.0
        'xxhash>=3.0.0', # allow later versions, tested with 3.1.0
        'blis>=0.7.5', # allow later versions, tested with 0.7.9
        'click>=8.0.4', # allow later versions, tested with 8.1.3
        'pydantic>=1.10.0,<2.0', # for spacy compatibility; avoid 2.0 due to breaking changes
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
