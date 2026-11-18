# SKA SDP Instrumental Calibration Pipeline

[![Documentation Status](https://readthedocs.org/projects/ska-telescope-ska-sdp-instrumental-calibration/badge/?version=latest)](https://ska-telescope-ska-sdp-instrumental-calibration.readthedocs.io/en/latest/?badge=latest)

The **Instrumental Calibration Pipeline (INST)** is a CLI application which generates the initial calibration solutions during standard SKA batch processing. It predicts model visibilities from a sky model (the GLEAM catalogue or an SKA LSM CSV file), and solves for gains by comparing them with the observed visibilities. It provides three pipelines: the instrumental calibration pipeline, `ska-sdp-instrumental-calibration`, which solves for the station delays and the bandpass on a calibrator, and two target calibration pipelines, `ska-sdp-instrumental-target-calibration` and `ska-sdp-instrumental-target-ionospheric`, which solve for phase-only complex gains and for an ionospheric phase screen on the target. The solutions are exported as h5parm files. It is built on the [piper](https://gitlab.com/ska-telescope/sdp/ska-sdp-exec-piper) framework, runs with local threads or on a dask cluster, and is developed by Team DHRUVA for SKAO.

Refer to the [Pipeline Flow](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/pipeline_flow.html) page for the stages of each pipeline, the data they solve on, and example configs.

The repository is [hosted on gitlab](https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration).
The documentation is available [at this page](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/).

If you wish to contribute to this repository, please refer [Developer Guide](DEVELOPMENT.md)

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

The latest release is available in SKA's pip repository. You can install this package using following command:

```bash
pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple ska-sdp-instrumental-calibration
```

To also install the optional `dp3` extra:

```bash
pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple "ska-sdp-instrumental-calibration[dp3]"
```

### As spack package

The instrumental calibration pipeline is available as the `py-ska-sdp-instrumental-calibration` [spack](https://spack.readthedocs.io/) package, in the [ska-sdp-spack](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack) repository. Follow its [README](https://gitlab.com/ska-telescope/sdp/ska-sdp-spack/-/blob/main/README.md) to set up spack, then install and load the pipeline:

```bash
spack install py-ska-sdp-instrumental-calibration
spack load py-ska-sdp-instrumental-calibration
```

### As OCI container

The pipeline can also be deployed inside an OCI container.

1. Run following command to pull the OCI image.

    ```bash
    docker pull artefact.skao.int/ska-sdp-instrumental-calibration:2.0.0
    ```

    The entrypoint of above image is set to the executable `ska-sdp-instrumental-calibration`.

1. Run image with volume mounts to enable read write to storage.

    ```bash
    docker run [-v local:container] <image-name> [run | install-config | experimental] ...
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
                    --set parameters.bandpass_calibration.run_solver_config.solver normal_equations \
                    --set parameters.load_data.nchannels_per_chunk 64
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

Please run `ska-sdp-instrumental-calibration run --help` to see  all supported options of the `run` subcommand.

The pipeline can run with local threads, on a dask cluster it creates itself (locally or inside a SLURM allocation), or on an existing dask cluster. Please refer to the ["Running the Pipeline"](https://developer.skao.int/projects/ska-sdp-instrumental-calibration/en/latest/running_pipeline.html) section of the documentation.

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

## Utility tools

Installing the package also installs these command line tools:

| Command | Source | Description |
| --- | --- | --- |
| `oskar-to-ska-lsm` | `src/inst_cli_tools/oskar_to_ska_lsm.py` | Convert an OSKAR sky model file to an SKA LSM CSV. |
| `gleam-to-ska-lsm` | `src/inst_cli_tools/gleam_to_ska_lsm.py` | Convert the GLEAM-EGC catalogue to an SKA LSM CSV. |
| `ska-lsm-to-sourcedb` | `src/inst_cli_tools/ska_lsm_to_sourcedb.py` | Convert an SKA LSM CSV to a sourcedb sky model, as read by DP3. |
| `clean-with-gains` | `src/inst_cli_tools/clean-with-gains` | Apply a gain table (h5parm) to a measurement set with DP3, and image it with WSClean. Requires `DP3` and `wsclean`. |

Run any of them with `--help` to see its options.
