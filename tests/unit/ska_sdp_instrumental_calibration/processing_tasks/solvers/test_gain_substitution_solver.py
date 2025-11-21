from unittest.mock import patch

import numpy as np
import pytest

from ska_sdp_instrumental_calibration.processing_tasks.solvers import (
    gain_substitution_solver,
)

GainSubstitution = gain_substitution_solver.GainSubstitution


@pytest.fixture
def mock_data(generate_vis):
    """Create mock visibility data for testing."""
    vis, gaintable = generate_vis

    vis_vis = vis.vis.values
    vis_flags = vis.flags.values
    vis_weight = vis.weight.values

    ntime, nbaseline, nfreq, npol = vis_vis.shape
    model_vis = np.random.randn(
        ntime, nbaseline, nfreq, npol
    ) + 1j * np.random.randn(ntime, nbaseline, nfreq, npol)
    model_flags = np.zeros((ntime, nbaseline, nfreq, npol), dtype=bool)

    gain_gain = gaintable.gain.values
    gain_weight = gaintable.weight.values
    gain_residual = gaintable.residual.values

    ant1 = vis.antenna1.values
    ant2 = vis.antenna2.values

    return {
        "vis_vis": vis_vis,
        "vis_flags": vis_flags,
        "vis_weight": vis_weight,
        "model_vis": model_vis,
        "model_flags": model_flags,
        "gain_gain": gain_gain,
        "gain_weight": gain_weight,
        "gain_residual": gain_residual,
        "ant1": ant1,
        "ant2": ant2,
    }


def test_gain_substitution_initialization():
    """Test GainSubstitution initialization with default parameters."""
    solver = GainSubstitution()

    assert solver.refant == 0
    assert solver.phase_only is False
    assert solver.crosspol is False
    assert solver.niter == 50
    assert solver.tol == 1e-6


def test_gain_substitution_initialization_with_params():
    """Test GainSubstitution initialization with custom parameters."""
    solver = GainSubstitution(
        refant=2, phase_only=True, crosspol=True, niter=50, tol=1e-8
    )

    assert solver.refant == 2
    assert solver.phase_only is True
    assert solver.crosspol is True
    assert solver.niter == 50
    assert solver.tol == 1e-8


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.solvers."
    "gain_substitution_solver.create_point_vis"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.solvers."
    "gain_substitution_solver.gain_substitution"
)
def test_solve_should_perform_gain_substitution(
    mock_gain_substitution,
    mock_create_point_vis,
    generate_vis_mvis_gain_ndarray_data,
):
    mock_data = generate_vis_mvis_gain_ndarray_data

    pointvis_vis = np.random.randn(10, 6, 16, 4) + 1j * np.random.randn(
        10, 6, 16, 4
    )
    pointvis_weight = np.ones((10, 6, 16, 4))
    mock_create_point_vis.return_value = (pointvis_vis, pointvis_weight)

    expected_gain = mock_data["gain_gain"]
    expected_weight = mock_data["gain_weight"]
    expected_residual = mock_data["gain_residual"]
    mock_gain_substitution.return_value = (
        expected_gain,
        expected_weight,
        expected_residual,
    )

    solver = GainSubstitution(
        refant=0, phase_only=True, crosspol=False, niter=30, tol=1e-6
    )

    result = solver.solve(
        vis_vis=mock_data["vis_vis"],
        vis_flags=mock_data["vis_flags"],
        vis_weight=mock_data["vis_weight"],
        model_vis=mock_data["model_vis"],
        model_flags=mock_data["model_flags"],
        gain_gain=mock_data["gain_gain"],
        gain_weight=mock_data["gain_weight"],
        gain_residual=mock_data["gain_residual"],
        ant1=mock_data["ant1"],
        ant2=mock_data["ant2"],
    )

    # Verify create_point_vis was called
    mock_create_point_vis.assert_called_once_with(
        mock_data["vis_vis"],
        mock_data["vis_flags"],
        mock_data["vis_weight"],
        mock_data["model_vis"],
        mock_data["model_flags"],
    )

    # Verify gain_substitution was called with correct parameters
    mock_gain_substitution.assert_called_once_with(
        mock_data["gain_gain"],
        mock_data["gain_weight"],
        mock_data["gain_residual"],
        pointvis_vis,
        mock_data["vis_flags"],
        pointvis_weight,
        mock_data["ant1"],
        mock_data["ant2"],
        crosspol=False,
        niter=30,
        phase_only=True,
        tol=1e-6,
        refant=0,
    )

    assert result == (expected_gain, expected_weight, expected_residual)


@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.solvers."
    "gain_substitution_solver.create_point_vis"
)
@patch(
    "ska_sdp_instrumental_calibration.processing_tasks.solvers."
    "gain_substitution_solver.gain_substitution"
)
def test_solve_should_perform_gain_substitution_without_model_vis(
    mock_gain_substitution,
    mock_create_point_vis,
    generate_vis_mvis_gain_ndarray_data,
):
    mock_data = generate_vis_mvis_gain_ndarray_data
    pointvis_vis = mock_data["vis_vis"].copy()
    pointvis_weight = mock_data["vis_weight"].copy()
    mock_create_point_vis.return_value = (pointvis_vis, pointvis_weight)

    expected_gain = mock_data["gain_gain"].copy()
    expected_weight = mock_data["gain_weight"].copy()
    expected_residual = mock_data["gain_residual"].copy()
    mock_gain_substitution.return_value = (
        expected_gain,
        expected_weight,
        expected_residual,
    )

    solver = GainSubstitution(refant=1, phase_only=False, crosspol=True)

    result = solver.solve(
        vis_vis=mock_data["vis_vis"],
        vis_flags=mock_data["vis_flags"],
        vis_weight=mock_data["vis_weight"],
        model_vis=None,
        model_flags=None,
        gain_gain=mock_data["gain_gain"],
        gain_weight=mock_data["gain_weight"],
        gain_residual=mock_data["gain_residual"],
        ant1=mock_data["ant1"],
        ant2=mock_data["ant2"],
    )

    mock_create_point_vis.assert_called_once_with(
        mock_data["vis_vis"],
        mock_data["vis_flags"],
        mock_data["vis_weight"],
        None,
        None,
    )

    assert mock_gain_substitution.called
    assert result == (expected_gain, expected_weight, expected_residual)


def test_solve_raises_error_for_invalid_refant(
    generate_vis_mvis_gain_ndarray_data,
):
    mock_data = generate_vis_mvis_gain_ndarray_data

    solver = GainSubstitution(refant=211)  # Invalid refant (>= nants)

    with pytest.raises(ValueError) as err:
        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )

    assert str(err.value) == "gain_substitution: Invalid refant: 211"


def test_solve_raises_error_for_negative_refant(
    generate_vis_mvis_gain_ndarray_data,
):
    mock_data = generate_vis_mvis_gain_ndarray_data

    solver = GainSubstitution(refant=-1)

    with pytest.raises(ValueError) as err:
        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )
    assert str(err.value) == "gain_substitution: Invalid refant: -1"


def test_solve_raises_error_for_empty_model_vis(
    generate_vis_mvis_gain_ndarray_data,
):
    mock_data = generate_vis_mvis_gain_ndarray_data

    solver = GainSubstitution(refant=0)

    zero_model_vis = np.zeros_like(mock_data["model_vis"])

    with pytest.raises(ValueError, match="Model visibility is zero"):
        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=zero_model_vis,
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )


def test_solve_raises_error_for_mismatched_model_inputs(
    generate_vis_mvis_gain_ndarray_data,
):
    mock_data = generate_vis_mvis_gain_ndarray_data

    solver = GainSubstitution(refant=0)

    with pytest.raises(ValueError) as err:
        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=mock_data["model_vis"],
            model_flags=None,
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )
    expected_message = (
        "gain_substitution: model_vis and model_flags "
        "must both be provided or both be None"
    )
    assert str(err.value) == expected_message

    with pytest.raises(ValueError) as err:
        solver.solve(
            vis_vis=mock_data["vis_vis"],
            vis_flags=mock_data["vis_flags"],
            vis_weight=mock_data["vis_weight"],
            model_vis=None,
            model_flags=mock_data["model_flags"],
            gain_gain=mock_data["gain_gain"],
            gain_weight=mock_data["gain_weight"],
            gain_residual=mock_data["gain_residual"],
            ant1=mock_data["ant1"],
            ant2=mock_data["ant2"],
        )

    assert str(err.value) == expected_message
