import numpy as np
import pytest
from mock import ANY, MagicMock, patch

from ska_sdp_instrumental_calibration.stages import ionospheric_delay_stage
from ska_sdp_instrumental_calibration.xarray_processors import with_chunks


@pytest.fixture
def mock_upstream_output():
    mock_output = MagicMock(name="UpstreamOutput")
    mock_output.vis = MagicMock(name="original_vis")
    mock_output.modelvis = MagicMock(name="model_vis")
    mock_output.chunks = MagicMock(name="chunks")

    mock_output.__setitem__ = MagicMock()

    return mock_output


@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".IonosphericSolver"
)
def test_solver_runs_and_applies_correction(
    MockIonosphericSolver,
    mock_create_gaintable,
    mock_apply_gaintable,
    mock_upstream_output,
):

    mock_gaintable = MagicMock(name="gaintable")
    mock_corrected_vis = MagicMock(name="corrected_vis")
    chunked_mock_gaintable = MagicMock(name="chunked_gaintable")
    MockIonosphericSolver.solve.return_value = mock_gaintable
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
        plot_table=False,
        _output_dir_="OUTPUT_DIR",
    )

    MockIonosphericSolver.solve.assert_called_once_with(
        mock_upstream_output.vis,
        mock_upstream_output.modelvis,
        mock_create_gaintable.return_value,
        ANY,
        True,
        20,
        1e-6,
        None,
    )

    mock_create_gaintable.assert_called_once_with(
        mock_upstream_output.vis, "full", "B"
    )
    called_args, _ = MockIonosphericSolver.solve.call_args
    np.testing.assert_array_equal(called_args[3], np.array([0, 1, 0, 1]))

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
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".IonosphericSolver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".get_gaintables_path",
    return_value="/test/dir/output.h5parm",
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".get_plots_path",
    return_value="/test/dir/plot.png",
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".PlotGaintableFrequency"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.ionospheric_delay"
    ".dask.delayed",
    side_effect=lambda x: x,
)
def test_gaintable_export_is_triggered(
    mock_dask_delayed,
    mock_create_gaintable,
    mock_export_func,
    mock_plot_freq_func,
    mock_get_plot_path,
    mock_get_gaintable_path,
    MockIonosphericSolver,
    _,
    mock_upstream_output,
):
    """
    Tests that enabling `export_gaintable` correctly adds a Dask task.
    """
    mock_gaintable = MagicMock(name="gaintable_to_export")
    MockIonosphericSolver.solve.return_value = mock_gaintable
    mock_gaintable.pipe.return_value = mock_gaintable
    mock_plot_freq_func.return_value = mock_plot_freq_func

    ionospheric_delay_stage.stage_definition(
        mock_upstream_output,
        cluster_indexes=[0, 1, 0, 1],
        block_diagonal=True,
        niter=20,
        tol=1e-6,
        zernike_limit=None,
        export_gaintable=True,
        plot_table=True,
        _output_dir_="/test/dir",
    )

    mock_get_gaintable_path.assert_called_once_with(
        "/test/dir", "ionospheric_delay.gaintable.h5parm"
    )

    mock_get_plot_path.assert_called_once_with(
        "/test/dir", "ionospheric_delay"
    )

    mock_export_func.assert_called_once_with(
        mock_gaintable, "/test/dir/output.h5parm"
    )
    mock_plot_freq_func.assert_called_once_with(
        path_prefix="/test/dir/plot.png"
    )
    mock_plot_freq_func.plot.assert_called_once_with(
        mock_gaintable, figure_title="Ionospheric Delay", phase_only=True
    )

    assert mock_upstream_output.add_compute_tasks.call_count == 2
