# pylint: skip-file
# flake8: noqa

import numpy as np
import pytest
from distributed.utils_test import (
    cleanup,
    client,
    cluster_fixture,
    loop,
    loop_in_thread,
)
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.visibility.vis_io_ms import create_visibility_from_ms

from ska_sdp_instrumental_calibration.dask_wrappers.solver import run_solver
from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    load_ms,
    simplify_baselines_dim,
)


@pytest.mark.skip("Changed function signature")
def test_should_calculate_gaintable_from_visibitlies(generate_ms, client):
    ms_name = generate_ms
    # Read in the Vis dataset directly and generate gains
    vis = simplify_baselines_dim(create_visibility_from_ms(ms_name)[0])
    # Read in the Vis dataset in chunks and make a copy for the model
    fchunk = len(vis.frequency) // 2

    solution_interval = vis.time.data.max() - vis.time.data.min()
    gaintable = create_gaintable_from_visibility(
        vis, jones_type="B", timeslice=solution_interval
    )
    gaintable.gain.data = gaintable.gain.data * (
        np.random.normal(1, 0.1, gaintable.gain.shape)
        + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    chunkedvis = load_ms(ms_name, fchunk)
    chunkedmdl = chunkedvis.copy(deep=True)
    # Chunk the gains
    chunkedgt = gaintable.chunk({"frequency": fchunk})
    # Corrupt the vis with the gains
    chunkedvis = apply_gaintable_to_dataset(
        chunkedvis, chunkedgt, inverse=False
    )

    solvedgt = run_solver(vis=chunkedvis, modelvis=chunkedmdl)
    solvedgt = client.compute(solvedgt, sync=True)

    # Phase ref input data for comparisons
    gaintable.gain.data *= np.exp(
        -1j * np.angle(gaintable.gain.data[:, [0], :, :, :])
    )
    assert np.allclose(solvedgt.gain.data, gaintable.gain.data, atol=1e-6)
