# SKA SDP Instrumental Calibration Pipeline

[![Documentation Status](https://readthedocs.org/projects/ska-telescope-ska-sdp-instrumental-calibration/badge/?version=latest)](https://ska-telescope-ska-sdp-instrumental-calibration.readthedocs.io/en/latest/?badge=latest)

The **Instrumental Calibration Pipeline (INST)** is a cli application which is used
to perform calibration operations on the SKA visibility data. It mainly provides Instrumental calibration pipeline and Taget calibration pipeline.

This repository contains the functions to generate the
initial calibration products during standard SKA batch processing. It includes
processing functions to prepare, model and calibrate a visibility dataset, data
handling functions for parallel processing, and high level workflow scripts and
notebooks.

If you wish to contribute to this repository, please refer [Developer Guide](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/DEVELOPMENT.html)

## Requirements for running the pipeline

The INST pipeline depends on these astronomy libraries:

| Library | Used for |
| --- | --- |
| [python-casacore](https://github.com/casacore/python-casacore) | Reading Measurement Set tables and metadata. |
| [EveryBeam](https://git.astron.nl/RD/EveryBeam) | Station beam responses, used to predict model visibilities and to normalise at the beam centre. |
| [DP3](https://git.astron.nl/RD/DP3) (optional) | The `dp3_gaincal` solver, which runs the DP3 `gaincal` step. Installed with the `dp3` extra. |
| [ska-sdp-func-python](https://developer.skao.int/projects/ska-sdp-func-python/en/) | SKA processing functions: gain and ionospheric solvers, and applying gains to visibilities. |
| [ska-sdp-datamodels](https://developer.skao.int/projects/ska-sdp-datamodels/en/) | SKA standard data models for visibilities, gain tables and sky models. |

All of these except DP3 are installed with the pipeline using the [standard installation steps](#installing-the-pipeline). DP3 needs the `dp3` extra, see below.

For prediction of model visibilities, here are the pre-requisites:

 * The GLEAM extragalactic catalogue or a sky model CSV file in the
   [SKA LSM format](https://developer.skao.int/projects/ska-sdp-global-sky-model/en/latest/design/lsm-file-structure.html).
   This and other catalogues will
   soon be available via
   [global-sky-model](https://developer.skao.int/projects/ska-sdp-global-sky-model/en/),
   but at present a hard copy is needed for prediction of model visibilities. The
   gleamegc catalogue can be downloaded via FTP from
   [VizieR](https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/100).
   The LSM CSV must start with a `CATALOGUE_METADATA_*` header (at least
   `VERSION`, `CATALOGUE_NAME` and `UPLOADED_AT`), otherwise it fails to load.
   Use the `oskar-to-ska-lsm` or `gleam-to-ska-lsm` scripts to convert an
   [OSKAR](https://ska-telescope.gitlab.io/sim/oskar/sky_model/sky_model.html#sky-model-file-fixed-format)
   sky model or the GLEAM-EGC catalogue into this format.
 * A measurement set with appropriate metadata to initialise the everybeam beam models.
   An appropriate measurement set for basic tests can be downloaded using the
   [everybeam package](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-everybeam/)
   script `download_ms.sh`, but one will also be made available in this package.

## Installing the pipeline

### In python environments

It is always recommended to create a seperate python environment for the pipeline.
For that, you can use `conda` or [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# To create a virtual environment using uv
# This will be created in the `.venv` directory
uv venv --python 3.11 --seed

#To activate the environment
source .venv/bin/activate
```

#### Stable release from SKAO pip index (recommended)

Run the following command to install the latest stable release (1.5.0) of the pipeline from SKAO python artifact repository:

```bash
INST_VERSION=1.5.0

# if using uv, use `uv pip install ...`
pip install --extra-index-url "https://artefact.skao.int/repository/pypi-internal/simple" "ska-sdp-instrumental-calibration==$INST_VERSION"
```

#### Latest pipeline from git

Run the following command to install the latest pipeline from the `main` branch of the git repository

```bash
INST_BRANCH=main

# if using uv, use `uv pip install ...`
pip install --extra-index-url "https://artefact.skao.int/repository/pypi-internal/simple" "ska-sdp-instrumental-calibration@git+https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git@$INST_BRANCH"
```

The `dp3_gaincal` solver is not in the 1.5.0 release. To use it, install
from git with the optional `dp3` extra:

```bash
pip install --extra-index-url "https://artefact.skao.int/repository/pypi-internal/simple" "ska-sdp-instrumental-calibration[dp3]@git+https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git@$INST_BRANCH"
```

### As spack package

The INST pipeline is available as a [spack](https://spack.readthedocs.io/en/v0.23.1/) package, in the [ska-sdp-spack](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack) repository. Please follow the [README](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/main/README.md) to setup spack on your machine. Then install the e2e pipeline using this command:

```bash
INST_VERSION=1.5.0

spack install "py-ska-sdp-instrumental-calibration@$INST_VERSION"
```

Then load the spack package with this command

```bash
spack load "py-ska-sdp-instrumental-calibration"
```

### As OCI container

We also provide a OCI (docker) image which is hosted on the SKA Docker artifact repository.
To pull the docker image for the latest stable release, please run:

```bash
INST_VERSION=1.5.0
docker pull "artefact.skao.int/ska-sdp-instrumental-calibration:$INST_VERSION"
```

The entrypoint of above image is set to the executable `ska-sdp-instrumental-calibration`.

Run image with volume mounts to enable read write to storage.

```bash
docker run [-v local:container] <image-name> ...<cli_options>...
```

## Using the CLI for Instrumental Calibration pipeline

Once you install the pipeline, you should be able to access the pipeline cli with `ska-sdp-instrumental-calibration` command.

Running `ska-sdp-instrumental-calibration --help` should show following output:

```bash
usage: ska-sdp-instrumental-calibration [-h] {run,install-config,experimental} ...

positional arguments:
  {run,install-config,experimental}
    run                 Run the pipeline
    install-config      Installs the default config at --config-install-path
    experimental        Allows reordering of stages via additional config section

options:
  -h, --help            show this help message and exit
```

### Generating YAML config of the pipeline

The INST pipeline expects a YAML config file as one of the inputs, which defines the stages and their parameters.
The information about stages is present in the documentation

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

### Running the pipeline

Run the instrumental calibration pipeline using `run` subcommand.

Example:

```bash
ska-sdp-instrumental-calibration run \
--config /path/to/config \
--output /path/to/output/dir \
/path/to/ms
```

Please run `ska-sdp-instrumental-calibration run --help` to see  all supported options of the `run` subcommand.\

### Reordering stages in pipeline

The `experimental` subcommand runs the stages in an order given in the config, and can run a stage more than once. See [Reordering Stages (Experimental)](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/experimental_pipeline.html) for the config format and how the stages are named and configured.

## Using the CLI for Target Calibration pipeline

Instrumental Calibration Pipeline for Target has two CLIs `ska-sdp-instrumental-target-calibration` and `ska-sdp-instrumental-target-ionospheric` commands which performs complex gain and ionospheric delay correction on Target respectively.

Running `ska-sdp-instrumental-target-calibration --help` should show following output:

```bash
usage: ska-sdp-instrumental-target-calibration [-h] {run,install-config} ...

positional arguments:
  {run,install-config}
    run                 Run the pipeline
    install-config      Installs the default config at --config-install-path

options:
  -h, --help            show this help message and exit
```

Running `ska-sdp-instrumental-target-ionospheric --help` should show following output:

```bash
usage: ska-sdp-instrumental-target-ionospheric [-h] {run,install-config} ...

positional arguments:
  {run,install-config}
    run                 Run the pipeline
    install-config      Installs the default config at --config-install-path

options:
  -h, --help            show this help message and exit
```

> Generating YAML config of the pipeline and running the pipeline is same as instrumental calibration pipeline. Target calibration pipeline doesn't provide experimental configuration.

## Calibration flow

Refer to the [calibration flow](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/calibration_flow.html) page for the stages of each pipeline, the data they solve on, and example configs.


