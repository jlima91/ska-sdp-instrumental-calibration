import numpy as np
import pytest
from mock import patch

from ska_sdp_instrumental_calibration.data_managers.gaintable import (
    create_gaintable_from_visibility,
)
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.xarray_processors import (
    ionosphere_solvers,
)


def test_solve_for_ionosphere(generate_vis):
    vis, jones = generate_vis

    jones.gain.data[..., 0, 1] *= 0
    jones.gain.data[..., 1, 0] *= 0

    assert np.all(vis.vis.data[..., :] == [1, 0, 0, 1])
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)

    modelvis = vis.copy(deep=True)
    modelvis.vis.data[..., :] = [1, 0, 0, 1]
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", skip_default_chunk=True, timeslice="full"
    )

    phases = ionosphere_solvers.IonosphericSolver.solve(
        vis, modelvis, gaintable
    ).compute()
    assert phases.gain.shape == gaintable.gain.shape

    np.testing.assert_allclose(np.median(np.angle(phases.gain.data)), 0)


def test_should_set_correct_polarization(generate_vis):
    vis, jones = generate_vis
    modelvis = vis.copy(deep=True)
    modelvis.vis.data[..., :] = [1, 0, 0, 1]

    solver = ionosphere_solvers.IonosphericSolver(vis, modelvis)
    np.testing.assert_array_equal(solver.pols, [0, 3])

    with patch(
        "ska_sdp_instrumental_calibration.xarray_processors"
        ".ionosphere_solvers.np.argwhere",
        return_value=np.array([[4]]),
    ):
        vis.attrs["_polarisation_frame"] = "stokesI"
        solver = ionosphere_solvers.IonosphericSolver(vis, modelvis)
        np.testing.assert_array_equal(solver.pols, [4])

    with pytest.raises(
        ValueError, match="build_normal_equation: Unsupported polarisations"
    ):
        vis.attrs["_polarisation_frame"] = "circular"
        solver = ionosphere_solvers.IonosphericSolver(vis, modelvis)


def test_should_raise_exception_for_zero_model_vis(generate_vis):
    vis, jones = generate_vis
    modelvis = vis.copy(deep=True)
    modelvis.vis.data[..., :] = [0, 0, 0, 0]

    with pytest.raises(
        ValueError, match="solve_ionosphere: Model visibilities are zero"
    ):
        ionosphere_solvers.IonosphericSolver(vis, modelvis)


def test_should_raise_exception_for_antenna_missmatch(generate_vis):
    vis, jones = generate_vis
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", skip_default_chunk=True, timeslice="full"
    )

    modelvis = vis.copy(deep=True)
    modelvis.vis.data[..., :] = [1, 0, 0, 1]
    cluster_indexes = np.zeros(2)

    solver = ionosphere_solvers.IonosphericSolver(
        vis, modelvis, cluster_indexes=cluster_indexes
    )

    with pytest.raises(ValueError, match="cluster_indexes has wrong size 2"):
        solver._solve(gaintable)
