SKA Instrumental Calibration Pipeline
=====================================

[![Documentation Status](https://readthedocs.org/projects/ska-sdp-instrumental-calibration/badge/?version=latest)](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/?badge=latest)

Instrument calibration pipeline for the SKA SDP.

The INST pipeline project contains the functions and scripts needed to generate the
initial calibration products during standard SKA batch processing. It includes
processing functions to prepare, model and calibrate a visibility dataset, data
handling functions for parallel processing, and high level workflow scripts and
notebooks.

Requirements
------------

The system used for development needs to have Python 3, `pip` and Poetry installed.
It uses standard SKA processing functions in the
[func](https://developer.skao.int/projects/ska-sdp-func/en/) and
[func-python](https://developer.skao.int/projects/ska-sdp-func-python/en/)
repositories, and standard data models in the
[datamodels](https://developer.skao.int/projects/ska-sdp-datamodels/en/) repository.

For detailed package requirements, see `pyproject.toml`. This is the Poetry config file
to manage application dependencies. To install Poetry, use
`curl -sSL https://install.python-poetry.org | python3 -`.

**Always** use a virtual environment.
[Pipenv](https://pipenv.readthedocs.io/en/latest/) is now Python's officially
recommended method. You are encouraged to use your preferred environment isolation
(i.e. `pip`, `conda` or `pipenv`) while developing locally.

Testing
-------

This project uses [PyTest](https://pytest.org) as the testing framework.

 * Run tests with `make python-test`
 * Running the test creates the `htmlcov` folder
    - Inside this folder a rundown of the issues found will be accessible using the
      `index.html` file
 * Or run tests directly, e.g. `pytest -s tests/*.py`
 
Documentation
-------------

The documentation generator for this project is derived from SKA's
[SKA Developer Portal repository](https://github.com/ska-telescope/developer.skatelescope.org)

 * In order to build the documentation for this project, execute the following under
`./docs`:
```bash
$ make html
```
 * Or from the base directory:
```bash
$ make docs-build html
```
* The documentation be viewed by opening the file `./docs/build/html/index.html`

Development
-----------

 * All tests should pass before merging the code.
 * CI does the following code analysis:
    - `isort --profile black --line-length 79 --check-only  src tests/`
    - `black --exclude .+\.ipynb --line-length 79 --check  src tests/`
    - `flake8 --show-source --statistics --max-line-length 79  src tests/`
    - `pylint --max-line-length 79  src tests/`



