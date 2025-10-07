import numpy as np
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    load_ms,
)
from ska_sdp_instrumental_calibration.processing_tasks.calibrate import (
    target_solver,
)
from ska_sdp_instrumental_calibration.processing_tasks.calibration import (
    apply_gaintable,
)
from ska_sdp_instrumental_calibration.workflow.utils import with_chunks


def test_run_solver_for_target_calibration(generate_ms):
    ms_path = generate_ms
    vis_chunks = {
        "baselineid": -1,
        "polarisation": -1,
        "spatial": -1,
        "time": 1,
        "frequency": -1,
    }
    # Read in the Vis dataset directly and generate gains
    vis = load_ms(ms_path, fchunk=4)
    model_vis = vis.copy(deep=True)

    gaintable = create_gaintable_from_visibility(
        vis, jones_type="G", timeslice=None
    )

    gaintable.gain.data = gaintable.gain.data * np.exp(
        0 + np.random.normal(0, 0.1, gaintable.gain.shape) * 1j
    )

    vis = apply_gaintable(vis=vis, gt=gaintable, inverse=False)
    chunkedvis = vis.pipe(with_chunks, vis_chunks)
    chunkedmdl = model_vis.pipe(with_chunks, vis_chunks)

    initialtable = create_gaintable_from_visibility(
        vis, jones_type="G", timeslice=None
    )

    init_chunkedgt = initialtable.pipe(with_chunks, vis_chunks)
    chunkedgt = gaintable.pipe(with_chunks, vis_chunks)

    solvedgt = target_solver.run_solver(
        vis=chunkedvis,
        modelvis=chunkedmdl,
        gaintable=init_chunkedgt,
        jones_type="G",
        phase_only=True,
        timeslice=None,
    )

    solvedgt.load()

    # Phase ref input data for comparisons
    chunkedgt.gain.data *= np.exp(
        -1j * np.angle(chunkedgt.gain.data[:, [0], :, :, :])
    )
    assert np.allclose(solvedgt.gain.data, chunkedgt.gain.data, atol=1e-6)
