# Scripts to simulate SKA-LOW visibilities

This directory contains the scripts and configuration required to generate
SKA Low simulated visibilities using [OSKAR simulator](https://gitlab.com/ska-telescope/sim/oskar).

NOTE: The following commands and some defaults in the yaml files expect that all the necessary input files are present in the current working directory. So it is preferred to run those scripts from within `scripts/ska_low_sim/` folder. For ease of use, you can create a symlink to the `ska_low_sim` folder in your regular working directory, separate from INST's repo:

```bash
ln -s ska-sdp-instrumental-calibration/scripts/ska_low_sim/
```

## Typical workflow

All the scripts read the same `sim.yaml`. Edit it first, then run the scripts in this order (see [Setup](#setup) for the
AWS session and the Python environment):

```bash
source .venv/bin/activate

# 1. Download the missing input files referenced in sim.yaml (needs an AWS session)
python get_data.py sim.yaml

# 2. (Optional) Generate the gain table. Only needed if simulate_visibility.gaintable is the
#    output_gaintable of the generate_gaintable section, and that file does not exist yet.
python generate_gaintable.py sim.yaml

# 3. (Optional) Generate the TEC screen. Only needed if simulate_visibility.tec_screen is the
#    output_tec_screen of the generate_tec section, and that file does not exist yet.
python generate_tec.py sim.yaml

# 4. Simulate the visibilities
python simulate_visibility.py sim.yaml

# 5. Write the field name, scan intent and scan number of the target into the MS (on the host,
#    needs python-casacore)
python postprocess_visibility.py sim.yaml <output-dir>/visibility.scan-<number>.ms
```

Run `get_data.py` again whenever you change an input path in `sim.yaml` (e.g. a new telescope model or TEC screen).
Files which are already present are not downloaded again. `get_data.py` does not download the gain table written by
`generate_gaintable.py`, or the TEC screen written by `generate_tec.py`, so the order above works when they do not
exist yet.

## Details of the files

| Filename                                 | Description |
| ---------------------------------------- | ----------- |
| generate_gaintable.py                    | Generates synthetic gain tables (OSKAR `gain_model.h5` format) with bandpass, time variations, and optional RFI corruptions. |
| generate_tec.py                          | Generates a multi-layer ionospheric TEC screen (FITS, read by OSKAR) with ARatmospy. |
| get_data.py                              | Downloads the missing input files (telescope model, sky models, etc.) referenced in `sim.yaml`. |
| simulate_visibility.py                   | Wrapper over `lib/run_oskar.py` to run the full simulation workflow, inside Singularity or in the current environment. |
| postprocess_visibility.py                | Writes the field name, scan intent and scan number of the target (from `fields` in `sim.yaml`) into the simulated MS. |
| sim.yaml                                 | Main configuration file containing simulation and script-specific parameters. |
| lib/run_oskar.py                         | CLI application used to run OSKAR simulations (run by `simulate_visibility.py`, inside the OSKAR Singularity image or not). |
| lib/sim_config.py                        | Dataclasses which load and validate `sim.yaml`. Used by all the scripts which read it. |
| lib/gaintable/h5parm_from_oskar_gains.py | Converts an OSKAR gain table HDF5 file into a DP3-compatible h5parm file. |
| lib/gaintable/plot_oskar_gaintable.py    | Generates diagnostic plots (amplitude/phase vs frequency/time) from an OSKAR gain table. |
| data/SKA_Low_AA2_SP5175_spline_data.npz  | Precomputed spline fits for the SKA-LOW AA2 bandpass response, used during gain simulations (`generate_gaintable.bandpass_spline_file`). |
| data/                                    | Input data, downloaded by `get_data.py` or generated (see [Data layout](#data-layout)). |

### Data layout

By default (paths in `sim.yaml`), all the input data is under `data/`:

```
data/
├── OSKAR-2.13.0-Python3.sif                # simulate_visibility.oskar_sif
├── SKA_Low_AA2_SP5175_spline_data.npz      # generate_gaintable.bandpass_spline_file
├── cable-delays/                           # simulate_visibility.cable_delay
├── gain-tables/                            # generate_gaintable.output_gaintable (and its plots), simulate_visibility.gaintable
├── sky-models/                             # simulate_visibility.gleam_file, lotss_file, sky_model_file
├── tec/                                    # generate_tec.output_tec_screen, simulate_visibility.tec_screen
└── telescope-models/                       # simulate_visibility.tel_model
```

`get_data.py` downloads the missing files to the paths given in `sim.yaml`, so these folders are created as needed.

### Configuration file --- sim.yaml

This file is "the source of truth" for any simulations done using these scripts.

This file has a custom schema, with a common section for common parameters (frequency, time related) and then a separate section per script.
An example file with all possible parameters is given in the same folder, with name sim.yaml.

The schema is defined by the dataclasses in `lib/sim_config.py` (`SimConfig`, `SimulateVisibilityConfig`,
`GenerateGaintableConfig`, ...). Every script validates the file when it loads it, before doing anything else:

-   Unknown keys are an error, so a typo (e.g. `gleam_fiel`) is not silently ignored.
-   Values must have the right type (`true`/`false`, integer, number, string).
-   Dependent values are checked, e.g. start frequency < end frequency, `rfi_start_freq_hz` and
    `rfi_end_freq_hz` are given when `rfi: true`, and `station_type` / `element_taper` are valid OSKAR options.
    `generate_gaintable.py` also checks that the RFI band is within the simulated band, and that the outliers
    do not corrupt more stations or channels than simulated.

When you add a new parameter, add it to the matching dataclass in `lib/sim_config.py`, then use it in the script.


### Gain table generation --- generate_gaintable.py

This script can generate a gaintable with the following effects:

1. Bandpass: Response of the receptor.
2. Time variant effects for bandpass
3. Offsets in response per station
4. RFI corruptions for specific frequencies
5. Gain outliers for stations and frequencies.

The script takes `sim.yaml` file as input. User can modify the parameters in that file to enable/disable effects, change simulation size.
The bandpass shape is read from the spline fits in `generate_gaintable.bandpass_spline_file` (by default
`./data/SKA_Low_AA2_SP5175_spline_data.npz`, which is part of this folder). The spline file covers 50 to 350 MHz
on a 1 MHz grid, and `simulation_start_frequency_hz` and `simulation_end_frequency_hz` must be on that grid
(e.g. 50 MHz and 125 MHz), else the script raises an error.

The number of stations of the gain table is read from the telescope model `simulate_visibility.tel_model`: one
station per row of its top-level layout file (`layout.txt`, `layout_enu.txt`, `layout_ecef.txt` or
`layout_wgs84.txt`, the first found, as in OSKAR). So the telescope model must be present (see
[Getting data](#getting-data)), and changing `tel_model` is enough to simulate another array.

### TEC screen generation --- generate_tec.py

This script generates an ionospheric TEC screen with [ARatmospy](https://github.com/abhinavn17/ARatmospy): several
layers of turbulence (scale size, speed, direction and height of each layer), which flow and evolve with time. The sum
of the layers is rescaled to `target_rms_tecu`, and written as a FITS file which OSKAR reads as an external TEC screen
(`simulate_visibility.tec_screen`).

The `generate_tec` section of `sim.yaml` sets the screen parameters. The common section gives:

-   The duration: the screen covers `observing_time_mins`, with one screen every `sampling_time_sec` (or
    `generate_tec.time_interval_sec`).
-   The reference frequency: the centre of the band (`simulation_start_frequency_hz` to `simulation_end_frequency_hz`).
-   The field: the script checks that the screen is wide enough for the field of the first target (radius
    `simulate_visibility.field_radius_deg`, at transit), and prints a warning if not. OSKAR puts the screen at
    300 km, centred on the telescope.

The output is about `8 x (number of times) x (screen pixels)^2` bytes (0.6 GB for the defaults and a 1 minute
observation), so increase `time_interval_sec` for long observations.

### Visibility simulation --- simulate_visibility.py

The workflow of this script is:

-   Creates an output directory: `<scenario>-<timestamp>` in the current
    directory, or the path given with `--output-dir`.
-   Runs `lib/run_oskar.py`, inside Singularity if `oskar_sif` is set, else with the current Python (see
    [Running without Singularity](#running-without-singularity)). The log is written to `simulate_visibility_py.out`.
    `lib/run_oskar.py` reads all the simulation parameters (common section and `simulate_visibility` section) from
    `sim.yaml`, given with `--config`. The config file and input files outside the current directory are bound into the
    container with `-B`, so they can be anywhere on the host. The first target of the first field is simulated.
-   `lib/run_oskar.py` stages the inputs in `inputs/` in the output directory, as a record of the run: it copies
    the config file (keeping its name) and the custom sky model (`sky_model_file`), and links the TEC screen
    (`tec_screen`). It copies the telescope model (`tel_model`, a directory) as `<name>.custom`, links the gain table and cable
    delays into the copy, and simulates with the copy.
-   Optionally makes a dirty image with DP3 and wsclean.

**The telescope model must contain no corruptions.** The gain table and cable delays are applied only when set in
`sim.yaml` (`gaintable`, `cable_delay`), by linking them into the copy of the model as `gain_model.h5` and
`cable_length_error.txt`. If the model itself contains either file, `simulate_visibility.py` prints a warning, and
`lib/run_oskar.py` does not copy the file, so it is not applied. To use it, move it out of the model and set
it in `sim.yaml`. `get_data.py` downloads the model without these files.

#### Options in the `simulate_visibility` section of sim.yaml

All are optional, except `tel_model` (the telescope model directory). They are read by `simulate_visibility.py` and `lib/run_oskar.py`, and documented by the comments in
`sim.yaml`, above each key.

## Setup

### Getting data

All supporting data is stored in SKAO S3 buckets.

Start an AWS session:

```bash
aws-vault exec <your-config-name>
```

(Optional) increase S3 concurrency:

```bash
aws configure set default.s3.max_concurrent_requests 100
```

Download the input files referenced in `sim.yaml`:

```bash
python get_data.py sim.yaml
```

The script checks every input path of the `simulate_visibility` section of `sim.yaml` (the telescope model `tel_model`, and the files) and downloads
only the missing ones, to the paths given in `sim.yaml`. Run it again whenever you change an input path in
`sim.yaml`. To only see what is present, and where the missing files would come from:

```bash
python get_data.py sim.yaml --dry-run
```

| Key in sim.yaml                                                  | Downloaded from |
| ---------------------------------------------------------------- | ------------------------------------------------------------- |
| `simulate_visibility.tel_model`                                  | `<s3_prefix>/telescope-models/<name>/`, without `cable_length_error.txt` and `gain_model.h5` (the model must contain no corruptions) |
| `simulate_visibility.oskar_sif`                                  | OSKAR GitHub release, version taken from `OSKAR-<version>-Python3.sif` |
| `simulate_visibility.gleam_file`, `lotss_file`, `sky_model_file` | `<s3_prefix>/sky-models/<file name>` |
| `simulate_visibility.tec_screen`                                 | `<s3_prefix>/tec/<file name>` |
| `simulate_visibility.gaintable`                                  | `<s3_prefix>/gain-tables/<file name>`. Skipped if it is the `output_gaintable` of `generate_gaintable` |
| `simulate_visibility.cable_delay`                                | `cable_length_error.txt` of the telescope model in S3 |

`<s3_prefix>` is `get_data.s3_prefix` in `sim.yaml` (default `s3://skao-sdp-testdata/PI27-Low-G4`). Commented
(disabled) inputs are not downloaded. Needs `aws` (for S3) and `wget` (for the OSKAR image).

### Python environment setup

You can use the following commands to create a new virtual environment and install the dependencies needed to run all the Python scripts:

```bash
# Use 'uv' to create new venv
uv venv --python 3.11 --seed

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

`generate_tec.py` also needs ARatmospy, which is not a Python package. Clone the Python 3 fork next to `sim.yaml`
(the default `generate_tec.aratmospy_dir`), or set `aratmospy_dir` to another folder:

```bash
git clone https://github.com/abhinavn17/ARatmospy
```

## Gaintable generation

```bash
# Activate the environment
source .venv/bin/activate

# Create OSKAR gaintable with default cli options and store it to given path
python generate_gaintable.py sim.yaml
```

You can generate various plots (Amp vs Freq, Waterfall) on the generated gaintable using `lib/gaintable/plot_oskar_gaintable.py`:

```bash
python lib/gaintable/plot_oskar_gaintable.py <oskar_gaintable_path> [<output_dir>] [--sampling-time <seconds>]
```

You can also convert the generated oskar gaintable file into a "h5parm" file compatible with DP3:

```bash
python lib/gaintable/h5parm_from_oskar_gains.py sim.yaml <oskar_gaintable_path> [<output_h5parm>]
```

The time axis of the h5parm is the same as in the MS simulated by `simulate_visibility.py` with the same `sim.yaml`: the
observation is centred on the `transit_time` of the first target. An existing output file is replaced.

## TEC screen generation

```bash
# Activate the environment
source .venv/bin/activate

# Create the TEC screen generate_tec.output_tec_screen
python generate_tec.py sim.yaml
```

To use it in the simulation, set `tec_screen: *gen_tec_screen` in the `simulate_visibility` section of `sim.yaml`.
An existing output file is replaced.

## Visibility simulation

Once you modify `sim.yaml` according to your needs, run the following command:

```bash
python simulate_visibility.py sim.yaml
```

To write the outputs to a fixed folder instead of `<scenario>-<timestamp>`:

```bash
python simulate_visibility.py sim.yaml --output-dir <path>
```

If `<path>` already exists, it must be the output of a previous run (it
contains `simulate_visibility_py.out`), and it is deleted without confirmation.
Otherwise the script raises `FileExistsError`. It is not deleted if it contains the
current directory, the config file or an input of the `simulate_visibility` section
(e.g. `data/`, or a config file copied to `<path>/inputs/` by a previous run). A folder outside the current
directory is bound into the Singularity container with `-B`.

With `simulate_visibility.oskar_sif` set (the default in `sim.yaml`), this command needs `singularity` on the `PATH`. Run
`get_data.py` to download the image.

### Running without Singularity

If `simulate_visibility.oskar_sif` is commented out in `sim.yaml`, `simulate_visibility.py` runs `lib/run_oskar.py` directly, with the
same Python that runs `simulate_visibility.py` (so the same venv and `PYTHONPATH`). This environment needs:

-   The packages in `requirements.txt` (see [Python environment setup](#python-environment-setup)).
-   OSKAR (the C/C++ library, built with CUDA to use `use_gpus: true`) and its Python package `oskar`. See the
    [OSKAR install instructions](https://github.com/OxfordSKA/OSKAR). The Python package is built against the
    installed library, for example:

    ```bash
    # Only if OSKAR is not installed in a standard location
    export OSKAR_INC_DIR=<oskar-install-prefix>/include
    export OSKAR_LIB_DIR=<oskar-install-prefix>/lib

    uv pip install "git+https://github.com/OxfordSKA/OSKAR.git@2.13.0#subdirectory=python"
    ```

    Set `LD_LIBRARY_PATH` if the OSKAR library is not found at run time.

`simulate_visibility.py` checks that `oskar`, `numpy`, `astropy` and `yaml` can be imported before it starts. To check by hand:

```bash
python -c "import oskar; print(oskar.__version__)"
```

## Post-processing of the visibilities

OSKAR writes an empty field name, `OBSERVE_TARGET#ON_SOURCE` as the scan intent, and 0 as the scan number into the MS.
`postprocess_visibility.py` sets them from the target in the `fields` section of `sim.yaml`, updating the MS in place:

| MS column           | Value |
| ------------------- | ----- |
| `FIELD::NAME`       | `field_id` of the target. Default: the target name (e.g. `Cal1`) |
| `STATE::OBS_MODE`   | `scan_intent` of the target (e.g. `CALIBRATE_BANDPASS#ON_SOURCE`). Default: not changed |
| `SCAN_NUMBER`       | The number in the MS name `visibility.scan-<number>.ms`, which `lib/run_oskar.py` sets to `scan_id_start` of the target + the scan index (so `scan_id_start` for the single scan of `simulate_visibility.py`) |

```bash
python postprocess_visibility.py sim.yaml <output-dir>/visibility.scan-300.ms
```

The target is the first target of the first field (the one simulated by `simulate_visibility.py`), unless
`--field` and `--target` are given. `--scan-number` overrides the scan number from the MS name.

This script needs `python-casacore` (in `requirements.txt`), which is not in the OSKAR Singularity image, so run it
on the host, after `simulate_visibility.py`.
