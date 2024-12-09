Overview of Pipelines
=====================

Instrumental calibration pipelines will need to be flexible to be useful for
both Mid and Low, but also to adapt as the array assemblies and our knowledge
of the instruments progress. They will also likely need to be able to work on
either dedicated calibration scans or target fields. However a few basic
workflows have been set up for use in the early array assemblies and to test
new data models and data handling approaches. Also to assess performance and
scaling.

Pipelines will by default create their own Dask cluster, but can instead be
passed an existing cluster. See the
`INST CI page <https://confluence.skatelescope.org/pages/viewpage.action?pageId=294236884>`_
for an example with a dask_jobqueue SLURMCluster.

Bandpass calibration
--------------------

The
:py:func:`~ska_sdp_instrumental_calibration.workflow.pipelines.bandpass_calibration`
pipeline has the following steps. This is still very much in test mode,
with various parameters setup for the small test datasets.

 * If called without an input Measurement Set, create a small one using
   :py:func:`~ska_sdp_instrumental_calibration.workflow.utils.create_demo_ms`
   that:

    * Does not add visibility sample noise. This could be added, but has been
      left out for now to check for precise convergence.
    * Uses the GLEAM sky model and a common EveryBeam station beam model.
    * Adds complex Gaussian corruptions to station Jones matrices.
    * Writes to disk in MSv2 format.

 * Read the MSv2 data into Visibility dataset.\ :sup:`1`
 * Predict model visibilities.\ :sup:`1`
 * Do bandpass calibration.\ :sup:`1`
 * Apply calibration corrections to the corrupted dataset and check against
   the model dataset.\ :sup:`1`

\ :sup:`1` xarray dataset map_blocks() is used to distribute frequency
sub-bands across dask tasks.

The pipeline is demonstrated in three different ways in notebook
`demo_bpcal_pipeline.ipynb`. Once with only gain corruptions and a gain-only
solver for a simple user-defined sky model, then again using a sky model
generated automatically with GLEAM and EveryBeam, then again with gain and
leakage corruptions and a polarised solver.

Bandpass calibration with polarisation rotation
-----------------------------------------------

The
:py:func:`~ska_sdp_instrumental_calibration.workflow.pipelines.bandpass_polarisation`
pipeline is similar but has extra steps, including an intermediate,
post-calibration full-band fit for relative rotation of linear polarisation
between stations. With some testing and development this workflow may be a good
match for the needs of Low, but this initial version is also intended to show
some of the different options available. Other dedicated ionospheric solvers
are also available and will be demonstrated in other pipelines.

 * Function
   :py:func:`~ska_sdp_instrumental_calibration.workflow.utils.create_demo_ms`
   is called with gain and leakage Jones matrix corruptions, as well as
   matrix rotations that increase with wavelength squared and change across the
   array.
 * Read the MSv2 data into Visibility dataset.\ :sup:`1`
 * Predict model visibilities with no knowledge of the rotations.\ :sup:`1`
 * Do bandpass calibration.\ :sup:`1` A polarised solver is used, but for some
   channels it is not fully converging. It is likely that the solutions are
   converging, but to local minima due to the range of large rotations (need to
   check how the func-python solvers declare convergence). In any case, the
   solutions are good enough for subsequent full-band fits. And these can be
   used to redo calibration with better starting conditions.
 * Function
   :py:func:`~ska_sdp_instrumental_calibration.processing_tasks.post_processing.model_rotations`
   is used to fit for a Rotation Measure that models the relative station
   polarisation rotations that increase linearly with wavelength squared.
   An example of the RM spectrum produced for one station is show below.
   The model RMs are used generate a pure rotation Jones dataset that can
   be used to better initialise calibration.
 * Do bandpass calibration again, starting with the new model Jones matrices.\
   :sup:`1`
 * Apply calibration corrections to the corrupted dataset and check against
   the model dataset.\ :sup:`1`

\ :sup:`1` xarray dataset map_blocks() is used to distribute frequency
sub-bands across dask tasks.

The pipeline is demonstrated in notebook `demo_bppol_pipeline.ipynb`.

.. image:: img/bppol_rm.png
