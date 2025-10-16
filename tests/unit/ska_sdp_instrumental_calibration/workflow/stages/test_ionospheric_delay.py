import numpy as np
import pytest
from mock import ANY, MagicMock, patch

from ska_sdp_instrumental_calibration.workflow.stages import (
    ionospheric_delay_stage,
)
from ska_sdp_instrumental_calibration.workflow.utils import with_chunks


@pytest.fixture
def mock_upstream_output():
    mock_output = MagicMock(name="UpstreamOutput")
    mock_output.vis = MagicMock(name="original_vis")
    mock_output.modelvis = MagicMock(name="model_vis")
    mock_output.chunks = MagicMock(name="chunks")

    mock_output.__setitem__ = MagicMock()

    return mock_output


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".IonosphericSolver"
)
def test_solver_runs_and_applies_correction(
    MockIonosphericSolver, mock_apply_gaintable, mock_upstream_output
):

    mock_gaintable = MagicMock(name="gaintable")
    mock_corrected_vis = MagicMock(name="corrected_vis")
    chunked_mock_gaintable = MagicMock(name="chunked_gaintable")
    mock_solver_instance = MockIonosphericSolver.return_value
    mock_solver_instance.solve.return_value = mock_gaintable
    mock_gaintable.pipe.return_value = chunked_mock_gaintable
    mock_apply_gaintable.return_value = mock_corrected_vis

    result = ionospheric_delay_stage.stage_definition(
        mock_upstream_output,
        cluster_indexes=[0, 1, 0, 1],
        block_diagonal=True,
        niter=20,
        tol=1e-6,
        zernike_limit=None,
        export_gaintable=False,
        _output_dir_="OUTPUT_DIR",
    )

    MockIonosphericSolver.assert_called_once_with(
        mock_upstream_output.vis,
        mock_upstream_output.modelvis,
        ANY,
        True,
        20,
        1e-6,
        None,
    )

    called_args, _ = MockIonosphericSolver.call_args
    np.testing.assert_array_equal(called_args[2], np.array([0, 1, 0, 1]))

    mock_solver_instance.solve.assert_called_once()

    mock_gaintable.pipe.assert_called_once_with(
        with_chunks, mock_upstream_output.chunks
    )
    mock_apply_gaintable.assert_called_once_with(
        mock_upstream_output.vis, chunked_mock_gaintable, inverse=True
    )

    mock_upstream_output.__setitem__.assert_called_once_with(
        "vis", mock_corrected_vis
    )

    mock_upstream_output.add_checkpoint_key.assert_called_once_with("vis")

    assert result is mock_upstream_output


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".IonosphericSolver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".get_gaintables_path",
    return_value="/test/dir/output.h5parm",
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages.ionospheric_delay"
    ".dask.delayed",
    side_effect=lambda x: x,
)
def test_gaintable_export_is_triggered(
    mock_dask_delayed,
    mock_export_func,
    mock_get_path,
    MockIonosphericSolver,
    _,
    mock_upstream_output,
):
    """
    Tests that enabling `export_gaintable` correctly adds a Dask task.
    """
    mock_solver_instance = MockIonosphericSolver.return_value
    mock_gaintable = MagicMock(name="gaintable_to_export")
    mock_solver_instance.solve.return_value = mock_gaintable
    mock_gaintable.pipe.return_value = mock_gaintable

    ionospheric_delay_stage.stage_definition(
        mock_upstream_output,
        cluster_indexes=[0, 1, 0, 1],
        block_diagonal=True,
        niter=20,
        tol=1e-6,
        zernike_limit=None,
        export_gaintable=True,
        _output_dir_="/test/dir",
    )

    mock_get_path.assert_called_once_with(
        "/test/dir", "ionospheric_delay.gaintable.h5parm"
    )

    mock_export_func.assert_called_once_with(
        mock_gaintable, "/test/dir/output.h5parm"
    )

    mock_upstream_output.add_compute_tasks.assert_called_once()
