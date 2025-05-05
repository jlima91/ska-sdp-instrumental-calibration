###########
Change Log
###########

This project adheres to `Semantic Versioning <http://semver.org/>`_.

Main
****

Added
-----
* Changed the way everybeam is called and added normalisation for OSKAR datasets.
* Functionality to use a csv file to generate the sky model.

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
