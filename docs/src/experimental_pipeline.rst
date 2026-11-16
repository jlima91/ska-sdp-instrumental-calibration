Reordering Stages (Experimental)
================================

The ``experimental`` subcommand of the instrumental calibration pipeline
runs the stages in an order given in the config, instead of the default
order described in :doc:`calibration_flow`. It also allows a stage to run
more than once.

.. code-block:: bash

   ska-sdp-instrumental-calibration experimental \
       --config /path/to/config \
       --output /path/to/output/dir \
       /path/to/ms

The config controls both the order of the stages and the settings of each
stage. A ready-to-use example is
`calibrator_inst_experimental.yml <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/configs/calibrator_inst_experimental.yml>`_.

Configuration
-------------

.. code-block:: yaml

   global_parameters:
     experimental:
       pipeline:
         - load_data: {}
         - predict_vis:
             use_everybeam: true
             flux_limit: 2.0
             fov: 5.0
         - bandpass_calibration: {}
         - delay_calibration: {}
         - generate_channel_rm:
             run_solver_config:
               solver: normal_equations
               refant: 0
               niter: 30
         - delay_calibration: {}
         - export_gain_table: {}
   parameters:
     bandpass_calibration:
       plot_config:
         plot_table: true
         fixed_axis: false
       run_solver_config:
         solver: gain_substitution
         refant: 0
         niter: 10
     delay_calibration:
       oversample: 16
       plot_config:
         plot_table: false
         fixed_axis: false
     export_gain_table:
       file_name: inst.gaintable
       export_format: h5parm
       export_metadata: false
     load_data:
       nchannels_per_chunk: 32
   pipeline: {}

The list under ``global_parameters.experimental.pipeline`` sets the stages
and their order. The example above runs:

#. ``load_data``
#. ``predict_vis``
#. ``bandpass_calibration``
#. ``delay_calibration``
#. ``generate_channel_rm``
#. ``delay_calibration_1``
#. ``export_gain_table``

Only the listed stages run. The ``pipeline`` section is ignored and can be
left empty.

.. warning::

   The stage order is not validated. Check that the order makes sense,
   for example that ``predict_vis`` runs before any solver stage. The
   pipeline does not work unless ``load_data`` is the first stage.

Stage names
-----------

The first occurrence of a stage keeps its name. Each later occurrence is
renamed ``<stage-name>_<n>``, where ``n`` counts the repeats from 1, in
the order of the list. In the example, the second ``delay_calibration``
is renamed ``delay_calibration_1``. This gives every stage a unique name,
which is used with ``--set``.

Stage configuration
-------------------

The settings of a stage come from, highest precedence first:

#. The ``--set`` CLI option.
#. The stage's entry in ``global_parameters.experimental.pipeline``.
#. The ``parameters.<stage>`` section.
#. The defaults of the stage.

The stage's entry in the list and ``parameters.<stage>`` are not merged.
If the entry is not empty, it replaces ``parameters.<stage>`` for that
stage, and anything it leaves out takes the stage default. An empty entry
(``{}``) uses ``parameters.<stage>``.

A repeated stage with an empty entry uses the section of the original
stage name. In the example, ``delay_calibration_1`` uses
``parameters.delay_calibration``.

With ``--set``, use the renamed stage name for repeated stages, for
example:

.. code-block:: bash

   ska-sdp-instrumental-calibration experimental ... \
       --set parameters.delay_calibration_1.plot_config.plot_table true
