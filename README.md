SKA Instrumental Calibration Pipeline
=====================================

[![Documentation Status](https://readthedocs.org/projects/ska-telescope-ska-sdp-instrumental-calibration/badge/?version=latest)](https://ska-telescope-ska-sdp-instrumental-calibration.readthedocs.io/en/latest/?badge=latest)

Instrument calibration pipeline for the SKA SDP. See
[online documentation](https://ska-telescope-ska-sdp-instrumental-calibration.readthedocs.io/en/latest).

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

If `generate_lsm_from_gleamegc` and `predict_from_components` are used to generate
model visibilities for calibration, a number of external datasets will also be
required:

 * The GLEAM extragalactic catalogue is used to generate the local sky model. This and
   other catalogues will soon be available via
   [global-sky-model](https://developer.skao.int/projects/ska-sdp-global-sky-model/en/),
   but at present a hard copy is needed to use `processing_tasks.lsm_tmp`. The
   catalogue can be downloaded via FTP from
   [VizieR](https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/100).
 * A measurement set with appropriate metadata is needed to initialise the everybeam
   beam models. An appropriate measurement set can be downloaded using the
   [everybeam package](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-everybeam/)
   script `download_ms.sh`, but one will also be made available in this package.
 * The [everybeam coeffs](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-everybeam/-/tree/master/coeffs)
   directory is also needed to generate beam models. The directory path supplied to
   `predict_from_components` is used to set environment variable `EVERYBEAM_DATADIR`.

For detailed package requirements, see `pyproject.toml`. This is the Poetry config file
to manage application dependencies. To install Poetry, use:
```bash
$ curl -sSL https://install.python-poetry.org | python3 -
```

**Always** use a virtual environment.
[Pipenv](https://pipenv.readthedocs.io/en/latest/) is now Python's officially
recommended method. You are encouraged to use your preferred environment isolation
(i.e. `pip`, `conda` or `pipenv`) while developing locally.

Using the INST CLI
--------------------

### Install the CLI

After cloning the [repo](https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration) and setting up the environment, run `poetry install` command. This should install `ska-sdp-instrumental-calibration` cli command and it should be accessible in the current environment.

```bash
ska-sdp-instrumental-calibration --help                                                       (inst-cal-new) 
usage: ska-sdp-instrumental-calibration [-h] {run,install-config} ...

positional arguments:
  {run,install-config}
    run                 Run the pipeline
    install-config      Installs the default config at --config-install-path

options:
  -h, --help            show this help message and exit
```

As of now, this command consists of placeholder stages which will not yield any useful results. 

### Install the config

Install the default config YAML of the pipeline to a specific directory using the `install-config` subcommand.

```bash
ska-sdp-instrumental-calibration install-config --config-install-path path/to/dir
```

Parameters of the default configurations can be overridden

```bash
ska-sdp-instrumental-calibration install-config --config-install-path path/to/dir \
                    --set parameters.bandpass_calibration.flagging true \
                    --set parameters.load_data.fchunk 64
```

### Run the pipeline

Run the instrumental calibration pipeline using `run` subcommand.

Example:

```bash
ska-sdp-instrumental-calibration run \
--input /path/to/ms \
--config /path/to/config \
--output /path/to/output/dir
```

For all the options, run `ska-sdp-instrumental-calibration run --help`.

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

