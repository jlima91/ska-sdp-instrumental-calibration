Changelog
#########

This project adheres to `Semantic Versioning <http://semver.org/>`_.

Main
****

(Latest changes on main)

0.3.4
*****

Fixed
-----
* Fixed issue with dask.array.float32 targeted at spack builds. 

0.3.3
*****

Added
-----
* Apply gaintable on visibility.
* Export corrected visibility and model visibility.


0.3.0
*****

Added
-----
* Added Faraday Rotation into modular INST Pipeline.
* Introduced export_visibility stage which currently supports applying gaintable on visibilities.
* Added smooth gain solutions stage.
* Allow exporting intermediate gain solutions as h5parm file.

Changed
-------
* Improved differential Faraday rotation fits in model_rotations and associated updates to bandpass_polarisation.
* Made load_data stage as reorderable.
* Made model rotation dask compatible.
* Input configuration file schema is changed slightly for generate_channel_rm stage.

Fixed
-----
* Logging issue in LSM generation.


0.2.2
*****

Added
-----
* Make metadata generation optional.


0.2.1
*****

Added
-----
* Changed the way everybeam is called and added normalisation for OSKAR datasets.
* Functionality to use a csv file to generate the sky model.
* CLI installable using Piper.
* Instrument calibration pipeline as a collection of stages.
* Delay calibration stage.
* Ability to reorder stages.
* Generation of calibration plots.
* Generation of metadata file along with data products.
* Allow configuring csv based custom components.
* The way everybeam models are initialised for Low datasets has been updated.
* Normalisation has been simplified and updated for OSKAR simulations.

Fixed
-----
* Bug in pipeline_config for parameter fov.
* Bug in deconvolve_gaussian for circular Gaussian components.

0.1.6
*****

Added
-----
* Discard unused polarisation dimensions before writing H5Parm file.
* H5Parm calibration solution output.

0.1.5
*****

Fixed
-----
* After gaintable creation, run_solver resets the gaintable interval to include all times. This avoids a bug in create_gaintable_from_visibility.

Changed
-------
* Forced a single polarisation chunk during load_ms, to be consistent with other dimensions.

0.1.4
*****

Changed
-------
* Forced a single time chunk during load_ms.

0.1.3
*****

Fixed
-----
* Documentation badge.

0.1.2
*****

Changed
-------
* LSM Component elliptical Gaussian parameter names have been updated.
* LSM Component parameter Fint200 has been replaced with flux and ref_freq.
* Pipeline argument "dask_cluster" replaced with "dask_scheduler_address", which accepts the cluster IP rather the the object.
* Station-dependent beam models are extracted from EveryBeam.
* Simulations and tests have been reduced to AA1 (from AA2).

0.1.1
*****

Added
-----
* PipelineConfig class for pipelines.
* Pipelines can accept a user defined-dask cluster.
* Default values for a number of lsm Component variables.
* More options in create_demo_ms.
* .readthedocs.yaml file.

Changed
-------
* Pipelines can be called with a user-defined local sky model and the option of not using a beam model.
* Pipeline parameter eb_ms defaults to the input measurement set.
* A single baseline chunk is set in load_ms. This dimension requires modification and auto chunking can cause confusion.
* Pre-define work array in predict_from_components to avoid memory leak build up.
* Improvements to dask handling in dask_wrappers.

0.1.0
*****

Added
-----
* Demo pipelines and notebooks. Documentation and unit tests.
* Functions to handle data-model confusion during xarray dask operations.
* Support for elliptical Gaussian sky components during predict.
* General calibration tasks with ask-enabled ingest, predict, solve and apply options.
* Pulled in content from ska-python-skeleton.
* Empty Python project directory structure.
