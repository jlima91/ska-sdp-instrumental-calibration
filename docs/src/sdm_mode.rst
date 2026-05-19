SDM Mode
========

Overview
--------

The INST pipeline supports `Science Data Model <https://confluence.skatelescope.org/display/SWSI/Science+Data+Model>`_ (**SDM**), which integrates
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
  ``<sdm-path>/sky/<field_id>/sky_model.csv`` must exist.

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
