import setuptools
import shutil

with open("./README.md", "r") as fh:
    long_description = fh.read()

# make a copy of install requirements so that it gets distributed with the wheel
shutil.copy('install_requires.txt', 'medcat/install_requires.txt')

with open("install_requires.txt") as f:
    # read every line, strip quotes and comments
    dep_lines = [l.split("#")[0].replace("'", "").replace('"', "").strip() for l in f.readlines()]
    # remove comment-only (or empty) lines
    install_requires = [dep for dep in dep_lines if dep]


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
              'medcat.tokenizers', 'medcat.utils.meta_cat', 'medcat.pipeline', 'medcat.utils.ner', 'medcat.utils.relation_extraction',
              'medcat.utils.saving', 'medcat.utils.regression', 'medcat.stats'],
    python_requires='>=3.9', # 3.8 is EoL
    install_requires=install_requires,
    include_package_data=True,
    package_data={"medcat": ["install_requires.txt"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
