import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("./README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="medcat",
    version="0.4.0.2",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Concept annotation tool for Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/MedCAT",
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing', 'medcat.cogstack'],
    install_requires=[
        'numpy~=1.18',
        'pandas~=1.0',
        'gensim~=3.7',
        'spacy==2.2.4',
        'scipy~=1.4',
        'tokenizers~=0.8',
        'torch~=1.4.0',
        'torchvision~=0.5.0',
        'Flask~=1.1',
        'sklearn~=0.0',
        'elasticsearch==7.9.1',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
