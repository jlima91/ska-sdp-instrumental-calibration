Calibration Flow
================

This page describes how data flows through the stages of the INST
pipelines, which visibilities each solver stage uses, and the default
configuration of each pipeline.

INST provides three pipelines:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Pipeline
     - Command
     - Exported solutions
   * - Calibrator (bandpass)
     - ``ska-sdp-instrumental-calibration``
     - Bandpass gains
   * - Target complex gain
     - ``ska-sdp-instrumental-target-calibration``
     - Phase-only complex gains
   * - Target ionospheric
     - ``ska-sdp-instrumental-target-ionospheric``
     - Ionospheric phase screen gains

The parameters of every stage are described in
:doc:`stage_config`.

The calibrator pipeline is based on the
`script <https://gitlab.com/ska-telescope/ssc/ska-ssc-low-comm-tools/-/blob/main/src/low_comm_tools/sdp/inst_mitch.py?ref_type=heads>`_
used by the CommSci team to validate the AA0.5 data. Its initial solves
follow RCAL.

Visibilities passed between stages
----------------------------------

Each stage receives the output of the previous stage and passes on an
updated version of it. The visibility datasets in it are:

``vis``
   The observed visibilities read by ``load_data`` (or
   ``target_load_data``).
``modelvis``
   The model visibilities predicted by ``predict_vis`` from the local sky
   model, including the station beam when ``use_everybeam`` is true.
``beam_normalised_vis`` and ``beam_normalised_modelvis``
   ``vis`` and ``modelvis`` multiplied by the inverse of the beam response
   in the beam pointing direction (the beam centre). They are only
   produced when ``predict_vis`` normalises at the beam centre, see
   below.
``corrected_vis``
   ``vis`` with the current gain table applied. Only produced by
   ``export_visibilities`` when ``apply_gaintable_to_vis`` is true.

.. _beam-centre-normalisation:

Normalisation at the beam centre
--------------------------------

The station beam scales the visibilities by the beam response. Solving
on visibilities and model visibilities normalised at the beam centre
removes most of this scaling from the solutions, which helps the initial
solves to converge. This follows the initial diagonal solve of RCAL.

``predict_vis`` controls this with ``normalise_at_beam_centre``:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Value
     - ``beam_normalised_vis``
     - ``beam_normalised_modelvis``
   * - ``true`` (default)
     - Produced
     - Produced
   * - ``"vis"``
     - Produced
     - Not produced
   * - ``"modelvis"``
     - Not produced
     - Produced
   * - ``false``
     - Not produced
     - Not produced

Normalisation needs the EveryBeam model, so nothing is normalised when
``use_everybeam`` is false, whatever the value of
``normalise_at_beam_centre``.

.. _normalised-data-fallback:

.. important::

   A stage that "solves on the normalised data" picks each side on its
   own, and silently falls back to the un-normalised data when the
   normalised version is not produced:

   - vis: ``beam_normalised_vis`` if produced, otherwise ``vis``.
   - model vis: ``beam_normalised_modelvis`` if produced, otherwise
     ``modelvis``.

   So what these stages actually solve on depends on
   ``normalise_at_beam_centre`` and ``use_everybeam``:

   - ``true`` (with EveryBeam): normalised vis against normalised model
     vis.
   - ``"vis"`` or ``"modelvis"``: only one side is normalised, so the
     solutions absorb the beam centre response.
   - ``false``, or ``use_everybeam: false``: un-normalised vis against
     un-normalised model vis, the same as every other stage.

Calibrator pipeline
-------------------

The stages run in the order below. Stages that are off by default can be
turned on in the ``pipeline`` section of the config. Each solver stage can
also export its intermediate gain table.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "lineColor": "#3b4a5a", "primaryColor": "#c9d3df", "primaryBorderColor": "#3b4a5a", "primaryTextColor": "#102030"}}}%%
   flowchart LR
       ms[/"Measurement Set"/] --> load["load_data"]
       sky[/"Sky model<br/>(GLEAM or SKA LSM CSV)"/] --> predict
       load --> predict["predict_vis"]
       predict --> delay["delay_calibration"]
       delay --> bpinit["bandpass_initialisation"]
       bpinit --> bp["bandpass_calibration"]
       bp --> flag["flag_gain"]
       flag --> rm["generate_channel_rm"]
       rm --> smooth["smooth_gain_solution"]
       smooth --> expvis["export_visibilities"]
       expvis --> expgain["export_gain_table"]
       expvis --> msout[/"Visibilities (.ms)"/]
       expgain --> h5[/"gaintable.h5parm"/]

       classDef data fill:#f2cf73,stroke:#8a6d1c,color:#2d2300
       classDef stage fill:#c9d3df,stroke:#3b4a5a,color:#102030
       classDef norm fill:#8fbce6,stroke:#1f5f99,color:#0b2540
       classDef unnorm fill:#9fd196,stroke:#2e6b27,color:#102a0c
       classDef off stroke-dasharray:6 4
       class ms,sky,msout,h5 data
       class load,predict,flag,smooth,expvis,expgain stage
       class delay,bpinit norm
       class bp,rm unnorm
       class flag,rm,smooth,expvis off

.. _diagram-legend:

The diagrams on this page use these colours and borders:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Shape
     - Meaning
   * - Blue box
     - Solver stage that solves on the beam centre normalised data,
       when produced (see :ref:`fallback <normalised-data-fallback>`).
   * - Green box
     - Solver stage that solves on the un-normalised data.
   * - Grey box
     - Stage that does not solve for gains (loading, prediction,
       flagging, smoothing, export).
   * - Yellow parallelogram
     - Input or output file.
   * - Dashed border
     - Stage that is off by default.

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Stage
     - Default
     - Data used
   * - ``load_data``
     - on
     - Reads ``vis`` and creates an empty bandpass gain table.
   * - ``predict_vis``
     - on
     - Predicts ``modelvis`` and, if enabled, the beam centre normalised
       vis and model vis.
   * - ``delay_calibration``
     - on
     - Solves on the **normalised** vis and model vis
       (see :ref:`fallback <normalised-data-fallback>`).
   * - ``bandpass_initialisation``
     - on
     - Solves on the **normalised** vis and model vis
       (see :ref:`fallback <normalised-data-fallback>`).
   * - ``bandpass_calibration``
     - on
     - Solves on the **un-normalised** ``vis`` (or ``corrected_vis``, see
       ``visibility_key``) and ``modelvis``.
   * - ``flag_gain``
     - off
     - Flags the gain table. Uses no visibilities.
   * - ``generate_channel_rm``
     - off
     - Solves on the **un-normalised** ``vis`` (or ``corrected_vis``) and
       the re-predicted ``modelvis``.
   * - ``smooth_gain_solution``
     - off
     - Smooths the gain table. Uses no visibilities.
   * - ``export_visibilities``
     - off
     - Writes ``vis`` (or ``corrected_vis``) and ``modelvis`` to
       Measurement Sets.
   * - ``export_gain_table``
     - on
     - Writes the gain table.

Only the initial solves, ``delay_calibration`` and
``bandpass_initialisation``, use the beam centre normalised data. The
later solves use the un-normalised data, so the exported bandpass is the
physical gain, including the beam centre response.

Notes on the stages:

- ``delay_calibration`` applies the solved delays to ``vis``, and to
  ``beam_normalised_vis`` when it is produced, so the later stages work on
  delay corrected visibilities.
- When ``extract_delays_from_vis`` is false (the default),
  ``delay_calibration`` runs a full gain solve, which also initialises the
  bandpass. ``bandpass_initialisation`` then skips its own solve and only
  exports the gain table.
- ``generate_channel_rm`` fits a rotation measure per station to the
  current gain table and re-predicts ``modelvis`` with it. When
  ``beam_normalised_modelvis`` is produced, it is updated from the new
  ``modelvis``, so a later initial solve stays consistent.
- The ``experimental`` subcommand allows the calibration stages to be
  reordered and repeated, see :doc:`experimental_pipeline`.

Target pipelines
----------------

The target complex gain and target ionospheric pipelines differ only in
their solver stage. The colours follow the :ref:`legend <diagram-legend>`
above.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "lineColor": "#3b4a5a", "primaryColor": "#c9d3df", "primaryBorderColor": "#3b4a5a", "primaryTextColor": "#102030"}}}%%
   flowchart LR
       ms[/"Measurement Set"/] --> load["target_load_data"]
       sky[/"Sky model<br/>(GLEAM or SKA LSM CSV)"/] --> predict
       load --> predict["predict_vis"]
       predict --> cg["complex_gain_calibration<br/>(target pipeline)"]
       predict --> iono["ionospheric_delay<br/>(ionospheric pipeline)"]
       cg --> expgain["export_gain_table"]
       iono --> expgain
       expgain --> h5[/"gaintable.h5parm"/]

       classDef data fill:#f2cf73,stroke:#8a6d1c,color:#2d2300
       classDef stage fill:#c9d3df,stroke:#3b4a5a,color:#102030
       classDef norm fill:#8fbce6,stroke:#1f5f99,color:#0b2540
       classDef unnorm fill:#9fd196,stroke:#2e6b27,color:#102a0c
       classDef off stroke-dasharray:6 4
       class ms,sky,h5 data
       class load,predict,expgain stage
       class cg,iono norm

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Stage
     - Default
     - Data used
   * - ``target_load_data``
     - on
     - Reads ``vis`` and creates an empty gain table with ``timeslice``
       solution intervals.
   * - ``predict_vis``
     - on
     - Same stage as in the calibrator pipeline.
   * - ``complex_gain_calibration`` (target pipeline only)
     - on
     - Phase-only solve on the **normalised** vis and model vis
       (see :ref:`fallback <normalised-data-fallback>`).
   * - ``ionospheric_delay`` (ionospheric pipeline only)
     - on
     - Ionospheric solve on the **normalised** vis and model vis
       (see :ref:`fallback <normalised-data-fallback>`).
   * - ``export_gain_table``
     - on
     - Writes the gain table.

Both target solves use the beam centre normalised data when
``predict_vis`` produces it.

Configurations
--------------

The `configs <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/tree/main/configs>`_
directory of the repository holds a ready-to-use config for each
pipeline, with the default or sensible example values filled in:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Config
     - Pipeline
   * - `calibrator_inst_run.yml <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/configs/calibrator_inst_run.yml>`_
     - Calibrator pipeline, ``run`` subcommand.
   * - `calibrator_inst_experimental.yml <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/configs/calibrator_inst_experimental.yml>`_
     - Calibrator pipeline, ``experimental`` subcommand, with the stages
       reordered and repeated.
   * - `target_inst.yaml <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/configs/target_inst.yaml>`_
     - Target complex gain pipeline.
   * - `target_ionospheric.yaml <https://gitlab.com/ska-telescope/sdp/science-pipeline-workflows/ska-sdp-instrumental-calibration/-/blob/main/configs/target_ionospheric.yaml>`_
     - Target ionospheric pipeline.

The default config of a pipeline, with every parameter at its default
value, is written by its ``install-config`` subcommand, for example:

.. code-block:: bash

   ska-sdp-instrumental-calibration install-config \
       --config-install-path ./config
