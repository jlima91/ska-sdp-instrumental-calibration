#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibration tests for the ska-sdp-instrumental-calibration module."""

# flake8 does not seem to like the generate_vis pytest fixture
# flake8: noqa: F401

import numpy as np
import pytest
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_func_python.preprocessing.averaging import averaging_frequency

from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
    solve_bandpass,
)
from tests.test_utils import generate_vis


def test_apply_gaintable(generate_vis):
    """Test application and correction of Jones matrices."""
    vis, jones = generate_vis

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
    vis, jones = generate_vis

    # Remove leakage from the GainTable, leaving only gain corruptions
    jones.gain.data[..., 0, 1] *= 0
    jones.gain.data[..., 1, 0] *= 0

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


def test_solve_bandpass(generate_vis):
    """Test solve_bandpass with gain-only corruptions."""
    np.random.seed(int(1e9))
    vis, jones = generate_vis

    # Give the vis some structure
    assert np.all(vis.vis.data[..., :] == [1, 0, 0, 1])
    vis.vis.data += np.random.normal(0, 0.2, vis.vis.shape)
    vis.vis.data += np.random.normal(0, 0.2, vis.vis.shape) * 1j

    # Make vis and gain models containing any known information
    modelvis = vis.copy(deep=True)
    gain_table = jones.copy(deep=True)
    gain_table.gain.data[..., :, :] = [[1, 0], [0, 1]]

    # Apply the gains and leakage to the true vis
    vis = apply_gaintable(vis=vis, gt=jones, inverse=False)

    # Solve for the gains and leakage
    refant = 0
    solve_bandpass(
        vis=vis,
        modelvis=modelvis,
        gain_table=gain_table,
        solver="normal_equations",
        refant=refant,
    )

    # check solutions (after phase referencing)
    shape = jones.gain.shape
    jones.gain.data *= np.exp(
        -1j * np.angle(jones.gain.data[:, refant, :, 0, 0])
    ).reshape(shape[0], 1, shape[2], 1, 1)
    gain_table.gain.data *= np.exp(
        -1j * np.angle(gain_table.gain.data[:, refant, :, 0, 0])
    ).reshape(shape[0], 1, shape[2], 1, 1)
    assert np.allclose(jones.gain.data, gain_table.gain.data)

    # check vis after applying solutions
    vis = apply_gaintable(vis=vis, gt=gain_table, inverse=True)
    assert np.allclose(vis.vis.data, modelvis.vis.data)


def test_solve_bandpass_resolution(generate_vis):
    """Test gain-only solve_bandpass with different spectral resolution."""
    vis, _ = generate_vis
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


def _solve(vischunk, refant):

    if len(vischunk.frequency) > 0:

        # Set multiple views into the combined dataset to keep the solver happy
        vis = vischunk.drop_vars(
            ["gain", "antenna", "receptor1", "receptor2", "modelvis"]
        )
        modelvis = vischunk.drop_vars(
            ["gain", "antenna", "receptor1", "receptor2", "vis"]
        ).rename({"modelvis": "vis"})
        solution_interval = np.max(vis.time.data) - np.min(vis.time.data)

        # Create a gaintable wrapper for the gain data
        gaintable = create_gaintable_from_visibility(
            vis,
            jones_type="B",
            timeslice=solution_interval,
        )
        gaintable.gain.data = vischunk.gain.data

        # Call the solver
        solve_bandpass(
            vis=vis,
            modelvis=modelvis,
            gain_table=gaintable,
            refant=refant,
        )

    return vischunk
