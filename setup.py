import setuptools

with open("./README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="medcat",
    version="0.2.0.3",
    author="w-is-h",
    author_email="w.kraljevic@gmail.com",
    description="Concept annotation tool for Electronic Health Records",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/w-is-h/cat",
    packages=['medcat', 'medcat.utils', 'medcat.preprocessing'],
    install_requires=[
        'numpy',
        'pandas',
        'gensim',
        'spacy',
        'scipy',
        'scispacy==0.2.0',],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
