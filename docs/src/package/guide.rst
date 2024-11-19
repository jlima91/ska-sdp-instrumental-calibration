.. doctest-skip-all
.. _package-guide:

.. todo::
    - Review and improve dask handling.
    - Time and frequency averaging before calibration.
    - XRADIO / MSv4 data models.
    - LOFAR HDF5 calibration output.

**********
Public API
**********

This package is intended to be used along with other SKA SDP batch processing
pipelines to generate instrumental calibration solutions. It is expected to run
early in batch processing, between batch pre-processing and self calibration.

See the package README file for dependencies and installation instructions.

Functions
---------

.. automodule:: ska_sdp_instrumental_calibration.logger
    :members:

.. automodule:: ska_sdp_instrumental_calibration.processing_tasks.calibration
    :members:

.. automodule:: ska_sdp_instrumental_calibration.processing_tasks.lsm_tmp
    :members:

.. automodule:: ska_sdp_instrumental_calibration.processing_tasks.predict
    :members:

Classes
-------
.. autoclass:: ska_sdp_instrumental_calibration.processing_tasks.lsm_tmp.Component
    :noindex:
    :members:
