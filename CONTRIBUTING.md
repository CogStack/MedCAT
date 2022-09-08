# Contributing to MedCAT

First of all, we would like to thank you for considering contributing towards MedCAT!

Please consider the below a guideline. Best judgment should be used in situations where the guidelines are not clear.

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

If you have questions regarding this project, you are better off expressing them in our [discourse](https://discourse.cogstack.org/) rather than creating an issue on github.

## Background Information

Most of the relevant links to background information are listed in our [README](README.md).

## How to Contribute

There are many ways of contributing to the project
- Reporting issues
- Suggesting new features
- Contributing to code

The following subsections will go into a little more detail regarding each of the above.

## Reporting Issues

Some things to remember when reporting an issue:
- Describe the issue clearly
- Provide the steps to reproduce the issue (if possible)
- Describe the behaviour you observed
- Describe the behaviour you expected
- Include all relevant information, including (but not limited to)
  - Version of MedCAT used
  - Versions of dependencies used
  - Config file(s)
  - Database(s) used
  - Deployment environment

## Suggesting New Features

MedCAT is always looking to grow and provide new features.

Some things to remember when suggesting a new feature:
- Describe the new feature in detail
- Describe the benefits of this new feature

## Contributing to Code

When making changes to MedCAT, make sure you have the dependencies defined in [requirements-dev](requirements-dev.txt).

Please make sure the code additions are adequately tested and well documented.

Before submitting a pull request, please ensure that the changes satisfy following:
- There are no issues with types/mypy (run `python -m mypy --follow-imports=normal medcat` in the project root)
- There are no issues with flake8 (run `flake8 medcat` in the project root)
- All tests are successful (run `python -m unittest discover` in the project root)

