import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("./README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="medcat",
    version="1.0.33",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Concept annotation tool for Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/MedCAT",
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing', 'medcat.cogstack', 'medcat.ner', 'medcat.linking', 'medcat.datasets', 'medcat.deprecated'],
    install_requires=[
        'numpy~=1.20',
        'pandas~=1.0',
        'gensim~=3.8',
        'spacy==2.3.4',
        'scipy~=1.5',
        'transformers~=4.5.1',
        'torch~=1.8.1',
        'Flask~=1.1',
        'sklearn~=0.0',
        'elasticsearch~=7.10',
        'dill~=0.3.3',
        'datasets~=1.6.0',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
