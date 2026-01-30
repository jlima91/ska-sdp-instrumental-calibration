# Scripts to simulate SKA-LOW visibilities

This directory contains the scripts and configuration required to generate
SKA Low simulated visibilities using [OSKAR simulator](https://gitlab.com/ska-telescope/sim/oskar)

NOTE: The following commands and some defaults in the yaml files expect that all the necessary input files are present in the current working directory. So it is preferred to run those scripts from within `scripts/ska_low_sim/` folder. For ease of use, you can create a simlink to the `ska_low_sim` folder in your regular working directory, seperate from INST's repo:

```bash
ln -s ska-sdp-instrumental-calibration/scripts/ska_low_sim/
```

## Details of the files

| Filename                           | Description                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| generate_gaintable.py              | Generates synthetic gain tables (H5parm format) with bandpass, time variations, and optional RFI corruptions. |
| get_data.sh                        | Downloads telescope models, sky models, etc., required to simulate SKA-LOW visibilities.                      |
| run_sim.py                         | Wrapper over `run_oskar.py` to run the full simulation workflow inside Singularity.                           |
| sim.yaml                           | Main configuration file containing simulation and script-specific parameters.                                 |
| SKA_Low_AA2_SP5175_spline_data.npz | Precomputed spline fits for the SKA-LOW AA2 bandpass response, used during gain simulations.                  |
| utils/h5parm_from_oskar_gains.py   | Converts an OSKAR gain table HDF5 file into a DP3-compatible h5parm file.                                     |
| utils/plot_gaintable.py            | Generates diagnostic plots (amplitude/phase vs frequency/time) from an OSKAR gain table.                      |
| utils/run_oskar.py                 | CLI application used to run OSKAR simulations.                                                                |

### Configuration file --- sim.yaml

This file is "the source of truth" for any simulations done using these scripts.

This file has a custom schema, with a common section for common parameters (frequency, time related) and then script specifc seperate section.
An example file with all possible parameters is given in the same folder, with name sim.yaml


### Gain table generation --- generate_gaintable.py

This script can generate a gaintable with following effects:

1. Bandpass: Response of the receptor.
2. Time variant effects for bandpass
3. Offsets in response per stations
4. RFI corruptions for specific frequencies
5. Gain outliers for stations and frequencies.

The script takes `sim.yaml` file as input. User can modify the parameters in that file to enable/disable effects, change simulation size.

### Visibility simulation --- run_sim.py

Following is the workflow of this script:

-   Creates an output directory.
-   Links the GLEAM catalog, telescope model, and TEC screen.
-   Links the gain model and cable delays into the telescope model
    folder.
-   Runs OSKAR inside Singularity.

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

Download data:

```bash
bash get_data.sh
```

The data will be downloaded into the current directory.

### Python environment setup

You can use following commands to create a new virtual environment and install necessary dependencies to run the all python scripts:

```bash
# Use 'uv' to create new venv
uv venv --python 3.11 --seed

# Activate the environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Gaintable generation

```bash
# Activate the environment
source .venv/bin/activate

# Create OSKAR gaintable with default cli options and store it to given path
python generate_gaintable.py sim.yaml
```

You can generate various plots (Amp vs Freq, Waterfall) on the generated gaintable using the plot_gaintable.py  from utils:

```bash
python utils/plot_gaintable.py <oskar_gaintable_path>
```

You can also convert the generated oskar gaintable file into a "h5parm" file compatible with DP3:

```bash
python utils/h5parm_from_oskar_gains.py sim.yaml <oskar_gaintable_path>
```

### Visibility simulation

Once you modify `sim.yaml` according to your needs, run the following command:

```bash
python run_sim.py sim.yaml
```

This command assumes that `singularity` is alread available.
