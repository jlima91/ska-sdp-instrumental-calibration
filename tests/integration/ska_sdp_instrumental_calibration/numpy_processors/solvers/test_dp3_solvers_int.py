import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.sky_model import SkyComponent
from ska_sdp_func_python.imaging.dft import dft_skycomponent_visibility

from ska_sdp_instrumental_calibration.numpy_processors.solvers import (
    dp3_solvers,
)

Dp3GaincalSolver = dp3_solvers.Dp3GaincalSolver
DP3ObservationInfo = dp3_solvers.DP3ObservationInfo
dp3_gaincal_solve = dp3_solvers.dp3_gaincal_solve

pytestmark = pytest.mark.requires_dp3


def _phase_reference(gain):
    """Reference the phases of all antennas to the first antenna."""
    return gain * np.exp(-1j * np.angle(gain[:, [0], :, :, :]))


def test_should_solve_gain_for_phase_only_disabled(
    generate_vis, apply_gaintable_to_dataset
):
    vis, _ = generate_vis

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

    # Distinct weights for the cross terms, which DP3 does not solve
    gain_weight = original_gaintable["weight"].values.copy()
    gain_weight[..., 0, 1] = 0.25
    gain_weight[..., 1, 0] = 0.25

    solver = Dp3GaincalSolver(niter=200, tol=1e-10)

    gain, weight, _ = solver.solve(
        vis_vis=vis.vis.values,
        vis_flags=vis.flags.values,
        vis_weight=vis.weight.values,
        model_vis=modelvis.vis.values,
        model_flags=modelvis.flags.values,
        gain_gain=original_gaintable["gain"].values,
        gain_weight=gain_weight,
        gain_residual=original_gaintable["residual"].values,
        ant1=vis.antenna1.data,
        ant2=vis.antenna2.data,
    )

    # DP3 gaincal has no reference antenna, so phase ref both gains.
    # The cross terms are not solved and keep their initial (zero) value.
    np.testing.assert_allclose(
        _phase_reference(gain),
        _phase_reference(gaintable.gain.values),
        atol=1e-6,
    )
    # The cross terms also keep their initial weights
    np.testing.assert_array_equal(weight[..., 0, 1], 0.25)
    np.testing.assert_array_equal(weight[..., 1, 0], 0.25)
    np.testing.assert_array_equal(weight[..., 0, 0], 1.0)
    np.testing.assert_array_equal(weight[..., 1, 1], 1.0)


def test_should_solve_gain_for_phase_only_enabled(
    generate_vis, apply_gaintable_to_dataset
):
    vis, _ = generate_vis

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

    solver = Dp3GaincalSolver(niter=200, tol=1e-10, phase_only=True)

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

    np.testing.assert_allclose(
        _phase_reference(gain),
        _phase_reference(gaintable.gain.values),
        atol=1e-6,
    )


def test_should_solve_gain_for_crosspol_enabled(
    generate_vis, apply_gaintable_to_dataset
):
    vis, _ = generate_vis

    solution_interval = vis.time.data.max() - vis.time.data.min()

    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )

    original_gaintable = gaintable.copy(deep=True)

    gaintable.gain.data = gaintable.gain.data + (
        np.random.normal(0, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    modelvis = vis.copy(deep=True)

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=False)

    solver = Dp3GaincalSolver(niter=200, tol=1e-10, crosspol=True)

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

    # For an unpolarised model, full Jones solutions are only unique up
    # to a unitary matrix. So instead correct the data with the solved
    # gains, and check that this recovers the uncorrupted model
    solved_gaintable = original_gaintable.copy(deep=True)
    solved_gaintable.gain.data = gain
    corrected = apply_gaintable_to_dataset(vis, solved_gaintable, inverse=True)

    np.testing.assert_allclose(
        corrected.vis.values, modelvis.vis.values, atol=1e-6
    )


def test_should_solve_gain_per_time_and_frequency_interval(
    generate_vis, apply_gaintable_to_dataset
):
    vis, _ = generate_vis

    # One solution per timeslot, one solution across all channels
    gaintable = create_gaintable_from_visibility(vis, jones_type="G")
    assert gaintable.gain.shape[0] == vis.time.size
    assert gaintable.gain.shape[2] == 1

    original_gaintable = gaintable.copy(deep=True)

    gaintable.gain.data = gaintable.gain.data * (
        np.random.normal(1, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    modelvis = vis.copy(deep=True)

    # Apply the frequency-independent gains to every channel
    vis_gaintable = create_gaintable_from_visibility(
        vis, jones_type="B"
    ).compute()
    vis_gaintable.gain.data = np.broadcast_to(
        gaintable.gain.values, vis_gaintable.gain.shape
    ).copy()
    vis = apply_gaintable_to_dataset(vis, vis_gaintable, inverse=False)

    solver = Dp3GaincalSolver(niter=200, tol=1e-10)

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

    assert gain.shape == gaintable.gain.shape
    np.testing.assert_allclose(
        _phase_reference(gain),
        _phase_reference(gaintable.gain.values),
        atol=1e-6,
    )


def test_should_ignore_flagged_visibilities(
    generate_vis, apply_gaintable_to_dataset
):
    vis, _ = generate_vis

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

    # Corrupt some samples, and flag them in either the data or the model
    vis_flags = np.zeros(vis.vis.shape, dtype=bool)
    model_flags = np.zeros(vis.vis.shape, dtype=bool)
    vis_flags[0, :10] = True
    model_flags[1, 10:20] = True
    vis.vis.data[vis_flags | model_flags] = 100.0

    solver = Dp3GaincalSolver(niter=200, tol=1e-10)

    gain, _, _ = solver.solve(
        vis_vis=vis.vis.values,
        vis_flags=vis_flags,
        vis_weight=vis.weight.values,
        model_vis=modelvis.vis.values,
        model_flags=model_flags,
        gain_gain=original_gaintable["gain"].values,
        gain_weight=original_gaintable["weight"].values,
        gain_residual=original_gaintable["residual"].values,
        ant1=vis.antenna1.data,
        ant2=vis.antenna2.data,
    )

    np.testing.assert_allclose(
        _phase_reference(gain),
        _phase_reference(gaintable.gain.values),
        atol=1e-6,
    )


def test_should_solve_gain_with_skymodel_prediction(
    generate_vis, apply_gaintable_to_dataset, tmp_path
):
    vis, _ = generate_vis

    # An unpolarised point source away from the phase centre, so that the
    # DP3 prediction depends on the uvw coordinates and their sign
    component = SkyComponent(
        direction=SkyCoord(ra=2.0, dec=-25.0, unit="deg"),
        frequency=vis.frequency.data,
        name="src",
        flux=np.ones((vis.frequency.size, 4)) * [1, 0, 0, 1],
        polarisation_frame=PolarisationFrame("linear"),
        shape="Point",
    )
    vis = dft_skycomponent_visibility(vis, component)

    # DP3 predicts XX = YY = I for an unpolarised source
    skymodel_path = tmp_path / "skymodel.txt"
    skymodel_path.write_text(
        "FORMAT = Name, Type, Ra, Dec, I, "
        "ReferenceFrequency='150e6', SpectralIndex='[]'\n"
        "src,POINT,2deg,-25deg,1.0,,[]\n"
    )

    solution_interval = vis.time.data.max() - vis.time.data.min()

    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )

    original_gaintable = gaintable.copy(deep=True)

    gaintable.gain.data = gaintable.gain.data * (
        np.random.normal(1, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    vis = apply_gaintable_to_dataset(vis, gaintable, inverse=False)

    gain, _, _ = dp3_gaincal_solve(
        vis_vis=vis.vis.values,
        vis_flags=vis.flags.values,
        vis_weight=vis.weight.values,
        model_vis=None,
        model_flags=None,
        gain_gain=original_gaintable["gain"].values,
        gain_weight=original_gaintable["weight"].values,
        gain_residual=original_gaintable["residual"].values,
        ant1=vis.antenna1.data,
        ant2=vis.antenna2.data,
        niter=200,
        tol=1e-10,
        skymodel_path=str(skymodel_path),
        observation=DP3ObservationInfo.from_visibility(vis),
    )

    np.testing.assert_allclose(
        _phase_reference(gain),
        _phase_reference(gaintable.gain.values),
        atol=1e-6,
    )
