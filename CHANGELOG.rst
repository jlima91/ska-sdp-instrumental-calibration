###########
Change Log
###########

This project adheres to `Semantic Versioning <http://semver.org/>`_.

Main
****

Added
-----
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

Fixed
-----

Removed
-------

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
