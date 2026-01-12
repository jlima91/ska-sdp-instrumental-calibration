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

The INST pipeline is primarily dependent on these external astronomy related libraries:

1. [python-casacore](https://github.com/casacore/python-casacore)
2. [everybeam](https://git.astron.nl/RD/EveryBeam)

Apart from above, the pipeline uses standard SKA processing functions in the

1. [sdp-func-python](https://developer.skao.int/projects/ska-sdp-func-python/en/)
2. [sdp-func](https://developer.skao.int/projects/ska-sdp-func/en/) (optional)

and SKA standard data models in the
[ska-sdp-datamodels](https://developer.skao.int/projects/ska-sdp-datamodels/en/) repository.

All above dependencies are installed along with the pipeline using [standard installation steps](#installing-the-pipeline).

For prediction of model visibilities, here are the pre-requisites:

 * The GLEAM extragalactic catalogue or a [OSKAR](https://ska-telescope.gitlab.io/sim/oskar/sky_model/sky_model.html#sky-model-file-fixed-format) csv file. This and other catalogues will
   soon be available via
   [global-sky-model](https://developer.skao.int/projects/ska-sdp-global-sky-model/en/),
   but at present a hard copy is needed for prediction of model visibilities. The
   gleamegc catalogue can be downloaded via FTP from
   [VizieR](https://cdsarc.cds.unistra.fr/viz-bin/cat/VIII/100).
 * A measurement set with appropriate metadata to initialise the everybeam beam models.
   An appropriate measurement set for basic tests can be downloaded using the
   [everybeam package](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-everybeam/)
   script `download_ms.sh`, but one will also be made available in this package.
 * The [everybeam coeffs](https://gitlab.com/ska-telescope/sdp/ska-sdp-func-everybeam/-/tree/master/coeffs)
   directory is also needed to generate beam models. The directory path supplied to
   `predict_from_components` is used to set environment variable `EVERYBEAM_DATADIR`.

## Foreword: Dask distribution

> This section is inspired from the [batch-preprocessing pipeline](https://developer.skao.int/projects/ska-sdp-batch-preprocess/en/latest/pipelines.html#foreword-dask-distribution)

> This is only applicable with you run INST pipeline with a dask cluster (LocalCluster or SlurmCluster)

The INST pipeline in the `load_data` stage, converts the MSv2 into a Zarr file, and stores it in the `cache_directory` path.

During the testing, we have realised that its better to limit the number of parallel tasks that run during the conversion from MSv2 to Zarr,
so that each task can get enought memory.

The only reliable solution is to use [worker resources](https://distributed.dask.org/en/latest/resources.html#worker-resources).

The instrumental calibration assumes that all workers define a resource called `process`; each worker may hold 1 or more `process` resources.
Each task of the conversion is defined to use 1 `process` resource.
Thus each worker will only run `process` number of tasks at any time (parallel/concurrent using its threadpool.)

To define the reources when starting dask worker using the cli command:

```bash
dask worker <SCHEDULER_ADDRESS> <OPTIONS> --resources "process=1"
```

Or in a LocalCluster:

```python
cluster = LocalCluster(resources={'process': 1})
```

> ⚠️ Warning:
> If the process resource is not defined on any worker, the pipeline (or rather, the Dask scheduler) will hang indefinitely.

## Installing the pipeline

### In python environments

It is always recommended to create a seperate python environment for the pipeline.
For that, you can use `conda` or [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# To create a virtual environment using uv
# This will be created in the `.venv` directory
uv venv --python 3.10 --seed

#To activate the environment
source .venv/bin/activate
```

#### Stable release from SKAO pip index (recommended)

Run the following command to install the latest stable release (0.6.0) of the pipeline from SKAO python artifact repository:

```bash
INST_VERSION=0.6.0

# if using uv, use `uv pip install ...`
pip install --extra-index-url "https://artefact.skao.int/repository/pypi-internal/simple" "ska-sdp-instrumental-calibration[python-casacore,ska-sdp-func]==$INST_VERSION"
```

#### Latest pipeline from git

Run the following command to install the latest pipeline from the `main` branch of the git repository

```bash
INST_BRANCH=main

# if using uv, use `uv pip install ...`
pip install --extra-index-url "https://artefact.skao.int/repository/pypi-internal/simple" "ska-sdp-instrumental-calibration[python-casacore,ska-sdp-func]@git+https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration.git@$INST_BRANCH"
```

### As spack package

The INST pipeline is available as a [spack](https://spack.readthedocs.io/en/v0.23.1/) package, in the [ska-sdp-spack](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack) repository. Please follow the [README](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/main/README.md) to setup spack on your machine. Then install the e2e pipeline using this command:

```bash
INST_VERSION=0.6.0

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
INST_VERSION=0.6.0
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
--input /path/to/ms \
--config /path/to/config \
--output /path/to/output/dir
```

Please run `ska-sdp-instrumental-calibration run --help` to see  all supported options of the `run` subcommand.\

### Reordering stages in pipeline

Run the instrumental calibration pipeline using `experimental` subcommand, to provide alternate stage order than the default order.

Example:

```bash
ska-sdp-instrumental-calibration experimental \
--input /path/to/ms \
--config /path/to/config \
--output /path/to/output/dir
```

The configuration is used to control both the execution order and any additional settings for each stage. The `experimental` subcommand allows reuse of the same stage multiple times.

#### Experimental Configuration

```yaml
global_parameters:
  experimental:
    pipeline:
      - load_data: {}
      - predict_vis:
          beam_type: everybeam
          flux_limit: 2.0
          fov: 5.0
      - bandpass_calibration: {}
      - delay_calibration: {}
      - generate_channel_rm:
          run_solver_config:
            solver: normal_equations
            refant: 0
            niter: 30
      - delay_calibration: {}
      - export_gain_table: {}
parameters:
  bandpass_calibration:
    plot_config:
      plot_table: true
      fixed_axis: false
    run_solver_config:
      solver: gain_substitution
      refant: 0
      niter: 10
  delay_calibration:
    oversample: 16
    plot_config:
      plot_table: false
      fixed_axis: false
  export_gain_table:
    file_name: inst.gaintable
    export_format: h5parm
    export_metadata: false
  load_data:
    nchannels_per_chunk: 32
pipeline: {}
```

The pipeline defined under `global_parameters.experimental.pipeline` will be used to construct the execution pipeline. It will consist of the following stages in the order: (1) `load_data` (2) `predict_vis` (3) `bandpass_calibration` (4) `delay_calibration` (5) `generate_channel_rm` (6) `delay_calibration_1` and (7) `export_gain_table`. There is no stage specific validations done while constructing the execution order, hence the user should pay special attention to stage order. The pipeline would not function if `load_data` stage is not set as the first stage.

**Stage Names**: The `ska-sdp-instrumental-calibration experimental` feature will update the stage names for duplicated stages as follows: the first occurrence of a stage name will remain unchanged, without a suffix. Subsequent duplicates will be renamed using the format `<stage-name>_x`, where `x` is the duplicate index starting from 1. For example, the second occurrence of `delay_calibration` will be renamed to `delay_calibration_1`. This numbering is automatically incremented for each duplicate, preserving the order as defined in the `global_parameters.experimental.pipeline` section. This approach ensures that each stage has a unique and identifiable name.

**The stage configurations** have the following precedence (from highest to lowest):

1.  `--set` cli parameter
2.  Configuration provided under the `global_parameters.experimental.pipeline.<stage>` section
3.  Configuration provided under `parameters.<stage>` section
4.  The default configurations used for the stage definitions.

While using the `--set` cli-option, please be mindful of the _suffix appended to the stage name_. Example: `ska-sdp-instrumental-calibration experimental ... --set parameters.delay_calibration_1.plot_config.plot_table true`

Please note that the `pipeline` section is intentionally left blank and would be ignored for the `ska-sdp-instrumental-calibration experimental` feature, as the stage execution order is decided from `global_parameters.experimental.pipeline` section.


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

## Calibation Strategy for Calibrator

Refer this page for [calibration strategy](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/calibration_strategy.html)


