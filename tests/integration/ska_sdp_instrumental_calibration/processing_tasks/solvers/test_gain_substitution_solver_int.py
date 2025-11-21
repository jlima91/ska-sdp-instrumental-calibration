import numpy as np
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
)
from ska_sdp_instrumental_calibration.processing_tasks.solvers import (
    gain_substitution_solver,
)

GainSubstitution = gain_substitution_solver.GainSubstitution


def test_should_solve_gain_for_phase_only_disabled(generate_ms):
    ms_name = generate_ms
    # Read in the Vis dataset directly and generate gains
    vis = create_visibility_from_ms(ms_name)[0]

    solution_interval = vis.time.data.max() - vis.time.data.min()

    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )

    original_gaintable = gaintable.copy(deep=True)

    gaintable.gain.data = gaintable.gain.data * (
        np.random.normal(1, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )
    modelvis = vis.copy(deep=True)

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=False)

    solver = GainSubstitution(niter=200)

    gain, _, _ = solver.solve(
        vis_vis=vis.vis.values,
        vis_flags=vis.flags.values,
        vis_weight=vis.weight.values,
        model_vis=modelvis.vis.values,
        model_flags=modelvis.flags.values,
        gain_gain=original_gaintable["gain"].values,
        gain_weight=original_gaintable["weight"].values,
        gain_residual=original_gaintable["residual"].values,
        ant1=vis.antenna1.data,
        ant2=vis.antenna2.data,
    )

    # Phase ref input data for comparisons
    gaintable.gain.data *= np.exp(
        -1j * np.angle(gaintable.gain.data[:, [0], :, :, :])
    )
    np.testing.assert_allclose(gain, gaintable.gain.values, atol=1e-6)


def test_should_solve_gain_for_phase_only_enabled(generate_ms):
    ms_name = generate_ms
    # Read in the Vis dataset directly and generate gains
    vis = create_visibility_from_ms(ms_name)[0]

    solution_interval = vis.time.data.max() - vis.time.data.min()

    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )

    original_gaintable = gaintable.copy(deep=True)

    gaintable.gain.data = gaintable.gain.data * np.exp(
        0 + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    modelvis = vis.copy(deep=True)

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=False)

    solver = GainSubstitution(niter=200, phase_only=True)

    gain, _, _ = solver.solve(
        vis_vis=vis.vis.values,
        vis_flags=vis.flags.values,
        vis_weight=vis.weight.values,
        model_vis=modelvis.vis.values,
        model_flags=modelvis.flags.values,
        gain_gain=original_gaintable["gain"].values,
        gain_weight=original_gaintable["weight"].values,
        gain_residual=original_gaintable["residual"].values,
        ant1=vis.antenna1.data,
        ant2=vis.antenna2.data,
    )

    # Phase ref input data for comparisons
    gaintable.gain.data *= np.exp(
        -1j * np.angle(gaintable.gain.data[:, [0], :, :, :])
    )

    np.testing.assert_allclose(gain, gaintable.gain.values, atol=1e-6)
