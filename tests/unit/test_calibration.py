#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the ska_python_skeleton module."""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_create import create_visibility
from ska_sdp_func_python.preprocessing.averaging import averaging_frequency

from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
    solve_bandpass,
)


@pytest.fixture
def generate_vis():
    """Fixture to build Visibility and GainTable datasets."""
    # Create the Visibility dataset
    config = create_named_configuration("LOWBD2")
    AA2 = (
        np.concatenate(
            (
                345 + np.arange(6),  # S8-1:6
                351 + np.arange(4),  # S9-1:4
                429 + np.arange(6),  # S10-1:6
                447 + np.arange(4),  # S13-1:4
                459 + np.arange(4),  # S15-1:4
                465 + np.arange(4),  # S16-1:4
                375 + np.arange(4),  # N8-1:4
                381 + np.arange(4),  # N9-1:4
                471 + np.arange(4),  # N10-1:4
                489 + np.arange(4),  # N13-1:4
                501 + np.arange(4),  # N15-1:4
                507 + np.arange(4),  # N16-1:4
                315 + np.arange(4),  # E8-1:4
                321 + np.arange(4),  # E9-1:4
                387 + np.arange(4),  # E10-1:4
                405 + np.arange(4),  # E13-1:4
            )
        )
        - 1
    )
    mask = np.isin(config.id.data, AA2)
    nstations = config.stations.shape[0]
    config = config.sel(indexers={"id": np.arange(nstations)[mask]})
    # Reset relevant station parameters
    nstations = config.stations.shape[0]
    config.stations.data = np.arange(nstations).astype("str")
    config = config.assign_coords(id=np.arange(nstations))
    # config.attrs["name"] = config.name+"-AA2"
    config.attrs["name"] = "AA2-Low-ECP-240228"
    vis = create_visibility(
        config=config,
        times=np.arange(3) * 0.9 / 3600 * np.pi / 12,
        frequency=150e6 + 1e6 * np.arange(4),
        channel_bandwidth=[1e6] * 4,
        phasecentre=SkyCoord(ra=0, dec=-27, unit="degree"),
        polarisation_frame=PolarisationFrame("linear"),
        weight=1.0,
    )
    # Put a point source at phase centre
    vis.vis.data[..., :] = [1, 0, 0, 1]

    return vis


def test_apply_gaintable(generate_vis):
    """Test application and correction of Jones matrices."""
    vis = generate_vis

    # Create the GainTable dataset
    jones = create_gaintable_from_visibility(vis, jones_type="B")
    jones.gain.data[..., 0, 0] = 1 - 0.1j
    jones.gain.data[..., 1, 1] = 3 + 0j
    jones.gain.data += np.random.normal(0, 0.2, jones.gain.shape)
    jones.gain.data += np.random.normal(0, 0.2, jones.gain.shape) * 1j

    # Apply the GainTable dataset to the Visibility dataset
    assert np.all(vis.vis.data[..., :] == [1, 0, 0, 1])
    # save a copy for testing
    orig = vis.copy(deep=True)
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)
    assert np.allclose(
        vis.vis.data,
        np.einsum(
            "...px,...xy,...qy->...pq",
            jones.gain.data[:, vis.antenna1.data, :, :, :],
            orig.vis.data.reshape(vis.vis.shape[:3] + (2, 2)),
            jones.gain.data[:, vis.antenna2.data, :, :, :].conj(),
        ).reshape(vis.vis.shape),
    )
    vis = apply_gaintable(vis=vis, gt=jones, inverse=True)
    assert np.allclose(vis.vis.data[..., :], [1, 0, 0, 1])


def test_solve_bandpass_unpolarised(generate_vis):
    """Test solve_bandpass with gain-only corruptions."""
    vis = generate_vis

    # Create the GainTable dataset with only gain corruptions
    jones = create_gaintable_from_visibility(vis, jones_type="B")
    g_sigma = 0.1
    jones.gain.data[..., 0, 0] = (
        np.random.normal(1, g_sigma, jones.gain.shape[:3])
        + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
    )
    jones.gain.data[..., 1, 1] = (
        np.random.normal(1, g_sigma, jones.gain.shape[:3])
        + np.random.normal(0, g_sigma, jones.gain.shape[:3]) * 1j
    )

    # Apply the GainTable dataset to the Visibility dataset
    assert np.all(vis.vis.data[..., :] == [1, 0, 0, 1])
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)

    # make copies with unit amplitude and zero phase
    modelvis = vis.copy(deep=True)
    modelvis.vis.data[..., :] = [1, 0, 0, 1]
    gain_table = jones.copy(deep=True)
    gain_table.gain.data[..., :, :] = [[1, 0], [0, 1]]
    # solve for new amplitudes and phases
    refant = 0
    solve_bandpass(
        vis=vis,
        modelvis=modelvis,
        gain_table=gain_table,
        refant=refant,
    )

    # check solutions (after phase referencing)
    jones.gain.data *= np.exp(
        -1j * np.angle(jones.gain.data[:, [refant], :, :, :])
    )
    assert np.allclose(jones.gain.data, gain_table.gain.data)

    # check vis after applying solutions
    vis = apply_gaintable(vis=vis, gt=gain_table, inverse=True)
    assert np.allclose(vis.vis.data, modelvis.vis.data)


def test_solve_bandpass_resolution(generate_vis):
    """Test gain-only solve_bandpass with different spectral resolution."""
    vis = generate_vis
    modelvis = vis.copy(deep=True)

    # Calibrate all-channel gaintable
    jones = create_gaintable_from_visibility(vis, jones_type="B")
    assert len(jones.frequency) == len(vis.frequency)
    modelvis.vis.data[..., :] = [1, 0, 0, 1]
    solve_bandpass(vis=vis, modelvis=modelvis, gain_table=jones)

    # Calibrate single-channel gaintable
    jones = create_gaintable_from_visibility(vis, jones_type="G")
    assert len(jones.frequency) == 1
    modelvis.vis.data[..., :] = [1, 0, 0, 1]
    solve_bandpass(vis=vis, modelvis=modelvis, gain_table=jones)

    # Calibrate erroneous gaintables
    with pytest.raises(ValueError):
        # wrong frequency value
        jones.frequency.data[0] += 1e3
        modelvis.vis.data[..., :] = [1, 0, 0, 1]
        solve_bandpass(vis=vis, modelvis=modelvis, gain_table=jones)
    with pytest.raises(ValueError):
        # wrong number of output channels
        modelvis.vis.data[..., :] = [1, 0, 0, 1]
        vis2 = averaging_frequency(vis, freqstep=2)
        jones2 = create_gaintable_from_visibility(vis2, jones_type="B")
        solve_bandpass(vis=vis, modelvis=modelvis, gain_table=jones2)
