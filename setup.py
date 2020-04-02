import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("./README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="medcat",
    version="0.3.2.4",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Concept annotation tool for Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/MedCAT",
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing'],
    install_requires=[
        'numpy~=1.18',
        'pandas~=1.0',
        'gensim~=3.7',
        'spacy==2.2.4',
        'scipy~=1.4',
        'tokenizers~=0.6.0',
        'torch~=1.4.0',
        'torchvision~=0.5.0',
        'Flask~=1.1',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
