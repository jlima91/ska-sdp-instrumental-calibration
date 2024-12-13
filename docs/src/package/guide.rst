.. doctest-skip-all
.. _package-guide:

.. todo::
    - Review and improve dask handling. See Vincent's comments in dask_wrappers.
    - Add a pipeline configuration passer and data class.
    - Time and frequency averaging before calibration.
    - Improve the interface to EveryBeam and loop over stations in
      GenericBeams.array_response.
    - XRADIO / MSv4 data models.
    - H5Parm calibration output.

**********
Public API
**********

This package will run as a batch processing pipeline (or pipelines), generating
instrumental calibration solutions. It will run early in batch processing for a given
observation, between batch pre-processing and self calibration.

See the package README file for dependencies and installation instructions.

Functions
---------

.. automodule:: ska_sdp_instrumental_calibration.logger
    :members:

.. automodule:: ska_sdp_instrumental_calibration.processing_tasks.calibration
    :members:

.. automodule:: ska_sdp_instrumental_calibration.processing_tasks.lsm
    :members:

.. automodule:: ska_sdp_instrumental_calibration.processing_tasks.predict
    :members:

Classes
-------
.. autoclass:: ska_sdp_instrumental_calibration.workflow.pipeline_config.PipelineConfig
    :noindex:
    :members:

.. autoclass:: ska_sdp_instrumental_calibration.processing_tasks.lsm.Component
    :noindex:
    :members:
