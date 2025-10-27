Changelog
#########

0.5.0
*****

Minor
-----

* Include script to apply gains and generate fits image using ``wsclean`` and ``dp3``
* Use XX = YY = 2I convention for predicting visibility from skycomponents
* Flagging based on the gain values
* Fix bug with plot labeling.
* Include pipeline validation notebooks for stage combinations

0.4.1
******

Patch
-----

* Fixed bug with incorrect beams computation

0.4.0
*****

Breaking
--------

* The configuration schema (YAML) has changed for many stages. Some notable changes:

  *  ``load_data`` stage now has parameters corresponding to the conversion from MSv2 to Zarr, like ``nchannels_per_chunk``, ``ntimes_per_ms_chunk``, ``cache_directory``.
  * Ineffective parameters like ``reset_vis``, ``jones_type``, ``export_model_vis``, ``flagging`` are removed.
  * ``fchunk`` parameter is removed, as we expect the entire pipeline to work with consistent chunksizes for all dimensions, from start till finish.

  Please refer to the :doc:`stage_config` page.

* For distributed run using dask workers, the workers must have a resource called ``process``. Please refer to the "dask distribution" section in :doc:`README<README>` to understand the usage.

Added
-----

* Support providing antenna names (along with indices) for config parameters which refer to a antenna, like ``refant`` or ``station``

Improvements
------------

Reducing memory footprint by using zarr as input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The pipeline will first convert the input MSv2 into a zarr file, which represents the ``Visibility`` data model.
* The zarr file will be chunked across frequency and time dimensions based on the parameters to ``load_data`` stage.
* Xarray operations like ``map_blocks`` work well with zarr format, minimising data which is loaded at a time to memory.
* The intermediate zarr files will be cached based on the name of MSv2, field id and data description id; and stored in user provided ``cache_directory``. This will ensure that cached zarr files are re-used between multiple runs on the same input MSv2.

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
