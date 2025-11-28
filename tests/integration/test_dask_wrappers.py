# flake8: noqa: F401
"""Test ska-sdp-instrumental-calibration wrappers around map_blocks calls."""

import numpy as np
import xarray as xr
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    load_ms,
    predict_vis,
    run_solver,
)
from ska_sdp_instrumental_calibration.numpy_processors.lsm import (
    Component,
    convert_model_to_skycomponents,
)
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.processing_tasks.predict import (
    predict_from_components,
)


def test_load_ms(generate_ms):
    ms_path = generate_ms
    # Read in the Visibility dataset directly
    vis = create_visibility_from_ms(ms_path)[0]

    # Read in the Visibility dataset in chunks
    fchunk = len(vis.frequency) // 2
    chunkedvis = load_ms(ms_path, fchunk)
    assert chunkedvis.chunks["frequency"][0] == fchunk

    # Check result
    chunkedvis.load()
    assert chunkedvis.frequency.equals(vis.frequency)
    # Can't compare these DataArrays directly because the baselines dim differs
    assert np.all(chunkedvis.vis.data == vis.vis.data)
    assert np.all(chunkedvis.weight.data == vis.weight.data)
    assert np.all(chunkedvis.flags.data == vis.flags.data)


def test_predict_vis(generate_ms):
    ms_path = generate_ms
    # Read in the Visibility dataset directly and predict a model
    vis = create_visibility_from_ms(ms_path)[0]
    modelvis = vis.assign({"vis": xr.zeros_like(vis.vis)})
    lsm = [
        Component(
            name="testcomp",
            RAdeg=vis.phasecentre.ra.degree - 0.3,
            DEdeg=vis.phasecentre.dec.degree + 1,
            flux=1.0,
            ref_freq=200e6,
            alpha=-0.5,
            major=0.2,
            minor=0.1,
            pa=10.0,
            beam_major=0.1,
            beam_minor=0.1,
            beam_pa=0.0,
        )
    ]
    # Evaluate LSM for current band
    lsm_components = convert_model_to_skycomponents(lsm, vis.frequency.data)
    # Call predict
    predict_from_components(
        modelvis,
        lsm_components,
        beam_type="not_everybeam",
        eb_coeffs=None,
        eb_ms=None,
    )

    # Read in the Visibility dataset in chunks and predict a model
    fchunk = len(vis.frequency) // 2
    chunkedvis = load_ms(ms_path, fchunk)
    chunkedmdl = predict_vis(
        chunkedvis,
        lsm,
        beam_type="not_everybeam",
    )
    assert chunkedvis.chunks["frequency"][0] == fchunk
    assert chunkedmdl.chunks["frequency"][0] == fchunk

    # Check result
    chunkedmdl.load()
    assert chunkedmdl.frequency.equals(modelvis.frequency)
    # Can't compare these DataArrays directly because the baselines dim differs
    assert np.allclose(chunkedmdl.vis.data, modelvis.vis.data)
    assert np.allclose(chunkedmdl.weight.data, modelvis.weight.data)
    assert np.allclose(chunkedmdl.flags.data, modelvis.flags.data)


def test_apply_gaintable(generate_ms):
    ms_path = generate_ms
    # Read in the Vis dataset directly and "correct" it with random gains
    vis = create_visibility_from_ms(ms_path)[0]
    solution_interval = vis.time.data.max() - vis.time.data.min()
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )
    gaintable.gain.data = gaintable.gain.data * (
        np.random.normal(1, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )
    vis = apply_gaintable(vis=vis, gt=gaintable, inverse=True)

    # Read in the Vis dataset in chunks and "correct" it with random gains
    fchunk = len(vis.frequency) // 2
    chunkedgt = gaintable.chunk({"frequency": fchunk})
    chunkedvis = load_ms(ms_path, fchunk)
    chunkedvis = apply_gaintable_to_dataset(
        chunkedvis, chunkedgt, inverse=True
    )
    assert chunkedvis.chunks["frequency"][0] == fchunk

    # Check result
    chunkedvis.load()
    assert chunkedvis.frequency.equals(vis.frequency)
    # Can't compare these DataArrays directly because the baselines dim differs
    assert np.all(chunkedvis.vis.data == vis.vis.data)
    assert np.all(chunkedvis.weight.data == vis.weight.data)
    assert np.all(chunkedvis.flags.data == vis.flags.data)


def test_run_solver(generate_ms):
    ms_path = generate_ms
    # Read in the Vis dataset directly and generate gains
    vis = create_visibility_from_ms(ms_path)[0]
    solution_interval = vis.time.data.max() - vis.time.data.min()
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )
    gaintable.gain.data = gaintable.gain.data * (
        np.random.normal(1, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    # Read in the Vis dataset in chunks and make a copy for the model
    fchunk = len(vis.frequency) // 2
    chunkedvis = load_ms(ms_path, fchunk)
    chunkedmdl = chunkedvis.copy(deep=True)
    # Chunk the gains
    chunkedgt = gaintable.chunk({"frequency": fchunk})
    # Corrupt the vis with the gains
    chunkedvis = apply_gaintable_to_dataset(
        chunkedvis, chunkedgt, inverse=False
    )
    assert chunkedvis.chunks["frequency"][0] == fchunk
    assert chunkedmdl.chunks["frequency"][0] == fchunk
    assert chunkedgt.chunks["frequency"][0] == fchunk

    solvedgt = run_solver(vis=chunkedvis, modelvis=chunkedmdl)

    solvedgt.load()

    # Phase ref input data for comparisons
    gaintable.gain.data *= np.exp(
        -1j * np.angle(gaintable.gain.data[:, [0], :, :, :])
    )
    assert np.allclose(solvedgt.gain.data, gaintable.gain.data, atol=1e-6)
