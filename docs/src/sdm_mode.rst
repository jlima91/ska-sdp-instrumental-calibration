SDM Mode
========

The INST pipelines can read their inputs from, and write their outputs to, a **Science Data Model (SDM)**: a shared directory through which the SDP pipelines exchange data. For example, the SKA SDP batch end-to-end pipeline creates an SDM, copies the sky models into it, and runs the INST pipelines on it, so that other pipelines can find their gain tables.

The SDM mode is turned on by passing ``--sdm-path``. Without it, the pipelines read the sky model given in the config, and write everything to the directory given with ``--output``.

Running with an SDM
-------------------

Pass the SDM directory with ``--sdm-path``, to any of the INST pipelines:

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yaml \
       --output /path/to/output_dir \
       --sdm-path /path/to/sdm_directory \
       input.ms

The ``run`` subcommands of ``ska-sdp-instrumental-target-calibration`` and ``ska-sdp-instrumental-target-ionospheric`` take the same option.

The SDM directory must already exist, with the sky model of the field in it (see `Expected layout`_). The pipeline fails at startup when the directory does not exist.

What changes with an SDM
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Data
     - Without ``--sdm-path``
     - With ``--sdm-path``
   * - Sky model, read by ``predict_vis``
     - ``gleamfile`` or ``lsm_csv_path`` from the config.
     - ``<sdm>/sky/<field>/sky_model.csv``. ``gleamfile`` and ``lsm_csv_path`` are ignored.
   * - Final gain table, written by ``export_gain_table``
     - ``<output>/<field>_gaintable.h5parm``
     - ``<sdm>/calibration/<purpose>/<field>/gaintable.h5parm``
   * - Logs, config, and QA products: intermediate gain tables, plots, the exported sky model, the dask performance report
     - ``<output>/``
     - ``<sdm>/logs/NN-inst/``, where ``NN`` is the next free index (``01``, ``02``, …), so every run gets its own directory.
   * - Zarr cache of the visibilities, and exported visibilities
     - ``<output>/``
     - ``<output>/`` (unchanged)

``<field>`` is the field ID, see `Field ID`_. The name of the gain table file is set with ``file_name`` in ``export_gain_table`` (by default ``gaintable``).

The ``<purpose>`` of the gain table depends on the pipeline:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Pipeline
     - Gain table in the SDM
   * - Calibrator (bandpass)
     - ``calibration/bandpass/<field>/gaintable.h5parm``
   * - Target complex gain
     - ``calibration/gains/<field>/gaintable.h5parm``
   * - Target ionospheric
     - ``calibration/ionosphere/<field>/gaintable.h5parm``

Field ID
--------

The field ID used in the SDM paths is the ``NAME`` of the field in the ``FIELD`` table of the input Measurement Set (its first row). When the name is empty, ``UNKNOWN_FIELD`` is used.

This is not the same as the ``field_id`` parameter of ``load_data``, which is the index of the field to read from the Measurement Set.

SDM layout
----------

Expected layout
^^^^^^^^^^^^^^^

Before a run, the SDM directory must hold the sky model of each field:

.. code-block:: text

   <sdm>/
   ├── sky/
   │   └── <field>/
   │       └── sky_model.csv
   ├── calibration/
   ├── logs/
   └── ...

.. _sdm-sky-model-format:

The ``sky_model.csv`` is read with ``ska-sdp-datamodels``, and must be in the SKA LSM format, starting with the catalogue metadata header lines, for example:

.. code-block:: text

   # (component_id,source_id,ra_deg,dec_deg,i_pol_jy,ref_freq_hz,epoch,a_arcsec,b_arcsec,pa_deg,spec_idx,log_spec_idx) = format
   # NUMBER_OF_COMPONENTS=1
   # CATALOGUE_METADATA_VERSION=1.0.0
   # CATALOGUE_METADATA_CATALOGUE_NAME=my_catalogue
   # CATALOGUE_METADATA_UPLOADED_AT=2026-01-01T00:00:00+00:00
   J000000-300000,,0.0,-30.0,1.5,200000000,0,0,0,0,"[-0.7]",true

A CSV without this header fails to load. The ``oskar-to-ska-lsm`` and ``gleam-to-ska-lsm`` scripts write it for you.

After a run
^^^^^^^^^^^

A run adds its gain table and a new logs directory to the SDM:

.. code-block:: text

   <sdm>/
   ├── calibration/
   │   └── <purpose>/
   │       └── <field>/
   │           └── gaintable.h5parm       # final gain table
   └── logs/
       └── NN-inst/                       # NN = next free index (01, 02, …)
           ├── <pipeline>_<time>.log      # pipeline log
           ├── <pipeline>_<time>.config.yaml
           ├── <pipeline>_<time>.cli.yaml
           ├── gaintables/                # intermediate gain tables
           ├── plots/
           └── sky/                       # exported sky model, if enabled

The ``gaintables``, ``plots`` and ``sky`` directories are only created when the stages export or plot something.
