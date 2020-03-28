import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

with open("./README.md", "r") as fh:
    long_description = fh.read()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        import subprocess
        import sys

        install.run(self)

        print("Installing the missing models for scispacy\n")
        pkg = 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz'
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])



setuptools.setup(
    name="medcat",
    version="0.3.1.1",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Concept annotation tool for Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CogStack/MedCAT",
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing'],
    install_requires=[
        'numpy~=1.15',
        'pandas~=0.23',
        'gensim~=3.7',
        'spacy==2.2.4',
        'scipy~=1.1',],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'install': PostInstallCommand,
        },

)
