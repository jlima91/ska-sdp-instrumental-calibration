import numpy as np
import pytest
from mock import ANY, MagicMock, patch

from ska_sdp_instrumental_calibration.stages.target_calibration import (
    ionospheric_delay_stage,
)
from ska_sdp_instrumental_calibration.xarray_processors import with_chunks


@pytest.fixture
def mock_upstream_output():
    mock_output = MagicMock(name="UpstreamOutput")
    mock_output.vis = MagicMock(name="original_vis")
    mock_output.modelvis = MagicMock(name="model_vis")
    mock_output.chunks = MagicMock(name="chunks")
    mock_output.timeslice = "timeslice"

    mock_output.__setitem__ = MagicMock()

    return mock_output


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "ionospheric_delay": {
            "cluster_indexes": None,
            "block_diagonal": True,
            "niter": 10,
            "tol": 1.0e-06,
            "zernike_limit": None,
            "plot_table": False,
        }
    }

    assert ionospheric_delay_stage.config == expected_config


def test_ionospeheric_delay_stage_is_mandatory():
    assert not ionospheric_delay_stage.is_optional


@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration."
    "ionospheric_delay.create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration."
    "ionospheric_delay.IonosphericSolver"
)
def test_solver_runs_and_updates_gaintable(
    MockIonosphericSolver,
    mock_create_gaintable,
    mock_upstream_output,
):

    mock_gaintable = MagicMock(name="gaintable")
    mock_initialtable = MagicMock(name="initialtable")
    chunked_mock_gaintable = MagicMock(name="chunked_gaintable")
    MockIonosphericSolver.solve.return_value = mock_gaintable
    mock_initialtable.pipe.return_value = chunked_mock_gaintable
    mock_create_gaintable.return_value = mock_initialtable

    result = ionospheric_delay_stage.stage_definition(
        mock_upstream_output,
        cluster_indexes=[0, 1, 0, 1],
        block_diagonal=True,
        niter=20,
        tol=1e-6,
        zernike_limit=None,
        plot_table=False,
        _output_dir_="OUTPUT_DIR",
    )

    MockIonosphericSolver.solve.assert_called_once_with(
        mock_upstream_output.vis,
        mock_upstream_output.modelvis,
        chunked_mock_gaintable,
        ANY,
        True,
        20,
        1e-6,
        None,
    )

    mock_create_gaintable.assert_called_once_with(
        mock_upstream_output.vis, "timeslice", "B", skip_default_chunk=True
    )

    mock_initialtable.pipe.assert_called_once_with(
        with_chunks, mock_upstream_output.chunks
    )

    called_args, _ = MockIonosphericSolver.solve.call_args
    np.testing.assert_array_equal(called_args[3], np.array([0, 1, 0, 1]))

    mock_upstream_output.__setitem__.assert_called_once_with(
        "gaintable", mock_gaintable
    )

    assert result is mock_upstream_output


@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration."
    "ionospheric_delay.PlotGaintableTargetIonosphere"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration."
    "ionospheric_delay.get_plots_path"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration."
    "ionospheric_delay.create_gaintable_from_visibility"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.target_calibration."
    "ionospheric_delay.IonosphericSolver"
)
def test_solver_runs_and_plots_gaintable(
    MockIonosphericSolver,
    mock_create_gaintable,
    get_plot_path_mock,
    plot_gain_target_iono_mock,
    mock_upstream_output,
):

    mock_gaintable = MagicMock(name="gaintable")
    mock_initialtable = MagicMock(name="initialtable")
    chunked_mock_gaintable = MagicMock(name="chunked_gaintable")
    MockIonosphericSolver.solve.return_value = mock_gaintable
    mock_initialtable.pipe.return_value = chunked_mock_gaintable
    mock_create_gaintable.return_value = mock_initialtable
    plot_gain_target_iono_mock.return_value = plot_gain_target_iono_mock

    ionospheric_delay_stage.stage_definition(
        mock_upstream_output,
        cluster_indexes=None,
        block_diagonal=True,
        niter=20,
        tol=1e-6,
        zernike_limit=None,
        plot_table=True,
        _output_dir_="OUTPUT_DIR",
    )

    get_plot_path_mock.assert_called_once_with(
        "OUTPUT_DIR", "ionospheric_delay"
    )

    plot_gain_target_iono_mock.assert_called_once_with(
        path_prefix=get_plot_path_mock.return_value
    )
    plot_gain_target_iono_mock.plot.assert_called_once_with(
        mock_gaintable, figure_title="Ionospheric", fixed_axis=True
    )

    mock_upstream_output.add_compute_tasks.assert_called_once_with(
        plot_gain_target_iono_mock.plot.return_value
    )
