SDM Mode
========

Overview
--------

The INST pipeline supports **Science Data Model(SDM)**, which integrates
with the SDP by exchanging data with other pipelines via a shared SDM
directory.

When ``--sdm-path`` is provided, the pipeline:

- Writes gaintables into the SDM directory under
  ``<sdm-path>/calibration/<purpose>/<field_id>/`` instead of the user provided output
  directory.
- Writes logs and QA products under a uniquely-indexed subdirectory
  ``<sdm-path>/logs/NN-inst/`` (where ``NN`` is the next available index).
- Resolves the sky model from the SDM directory using the field ID extracted
  from the input Measurement Set. The file
  ``<sdm-path>/sky/<field_id>/sky_model.csv`` must exist. It must be an SKA
  LSM CSV that starts with a ``CATALOGUE_METADATA_*`` header (see
  :ref:`sky model format <sdm-sky-model-format>`).

If you are running the pipeline as a standalone and do not pass
``--sdm-path``, pipeline works as usual, writing outputs and QA products in 
directory provided with ``--output``.

Gaintables will be written at given location for each pipeline:

- **Instrumental calibration**:  ``bandpass/<field_id>/gaintable.h5parm``
- **Instrumental target calibration**: ``gains/<field_id>/gaintable.h5parm``
- **Instrumental target ionospheric calibration**: ``ionosphere/<field_id>/gaintable.h5parm``


.. note::
  When ``--sdm-path`` and ``--output`` are used together, but ``--sdm-path`` takes
  precedence for gaintable, sky model and log output locations. When omitted,
  the pipeline falls back to writing outputs under ``--output``.


CLI Invocation
--------------

User can pass ``--sdm-path`` as follows while running any of the INST pipeline:

.. code-block:: bash

   ska-sdp-instrumental-calibration run \
       --config config.yaml \
       --output /path/to/output_dir \
       --sdm-path /path/to/sdm_directory \
       input.ms

It's similar for the target calibration and ionospheric calibration pipelines.


Directory Conventions
---------------------

When ``--sdm-path`` is provided, pipeline assumes following folder structure to
be present in SDM directory

.. code-block:: text

   <sdm>/
   ├── sky/
   │   └── <field_id>/
   │       └── sky_model.csv
   ├── calibration/
   ├── logs/
   └─ ...


.. _sdm-sky-model-format:

The ``sky_model.csv`` is read with ``ska-sdp-datamodels``, which requires the
catalogue metadata header lines before the components, for example:

.. code-block:: text

   # (component_id,source_id,ra_deg,dec_deg,i_pol_jy,ref_freq_hz,epoch,a_arcsec,b_arcsec,pa_deg,spec_idx,log_spec_idx) = format
   # NUMBER_OF_COMPONENTS=1
   # CATALOGUE_METADATA_VERSION=1.0.0
   # CATALOGUE_METADATA_CATALOGUE_NAME=my_catalogue
   # CATALOGUE_METADATA_UPLOADED_AT=2026-01-01T00:00:00+00:00
   J000000-300000,,0.0,-30.0,1.5,200000000,0,0,0,0,"[-0.7]",true

A CSV without this header fails to load. The ``oskar-to-ska-lsm`` and
``gleam-to-ska-lsm`` scripts write it for you.


Post-run, the pipeline updates the SDM directory like this:

.. code-block:: text

   <sdm>/
   ├── calibration/
   │   └── <purpose>/
   │       └── <field_id>/
   │           └── <filename>.h5parm     # exported gaintable
   └── logs/
       └── NN-inst/                      # NN = next available index (01, 02, …)
           ├── pipeline.log
           ├── config.yaml
           └── qa/
