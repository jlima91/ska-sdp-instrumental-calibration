#!/usr/bin/env python
"""
Aligned comparison of INST ``run_solver`` (dp3_ddecal) against DP3 execute.

Converted from the "#VINAY CODE" cell of
``notebooks/dp3_ddecal_load_ms_vinay_code1.ipynb``.

Both solvers are given a common model:

1. Copy the input MS and use DP3 ``predict`` to write the model into the
   ``MODEL_DATA`` column of the copy (the input MS is never modified).
2. Run DP3 DDECal via ``dp3.execute``, reading the model from
   ``MODEL_DATA`` (``cal.modeldatacolumns``).
3. Load ``DATA`` and ``MODEL_DATA`` of the copy into INST as ``vis`` and
   ``modelvis``, run ``run_solver`` with the ``dp3_ddecal`` solver, and
   compare its gains and weights with the DP3 execute h5parm.

Usage::

    python scripts/dp3_ddecal_aligned_comparison.py \
        [--ms INPUT.ms] [--skymodel SKY.skymodel] [--output-dir DIR] \
        [--use-model-column] [--scheduler tcp://HOST:PORT]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# DP3 refuses to run unless OpenBLAS is single-threaded, so set this
# before dp3 (and numpy) are imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# pylint: disable=wrong-import-position
import dask  # noqa: E402
import dp3  # noqa: E402
import numpy as np  # noqa: E402
from casacore.tables import table  # noqa: E402

from ska_sdp_instrumental_calibration.data_managers.gaintable import (  # noqa: E402,E501
    create_gaintable_from_h5parm,
)
from ska_sdp_instrumental_calibration.data_managers.solution_interval import (  # noqa: E402,E501
    SolutionIntervals,
)
from ska_sdp_instrumental_calibration.numpy_processors.solvers.solver import (  # noqa: E402,E501
    Solver,
)
from ska_sdp_instrumental_calibration.scheduler import (  # noqa: E402
    UpstreamOutput,
)
from ska_sdp_instrumental_calibration.stages.load_data import (  # noqa: E402
    load_data_stage,
)
from ska_sdp_instrumental_calibration.xarray_processors.solver import (  # noqa: E402,E501
    run_solver,
)

# compare_arrays lives in the notebooks folder, not in the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
from notebook_utils import compare_arrays  # noqa: E402

# pylint: enable=wrong-import-position

DEFAULT_MS_PATH = (
    "/shared/fsx1/users/dhruva/pi31/sbpt-21-inst-perf/preprocess/"
    "aa1_fix_avg.ms"
)
# The cell reuses dp3_skymodel_path from an earlier notebook cell
DEFAULT_SKYMODEL_PATH = str(
    Path(__file__).resolve().parents[1]
    / "notebooks/inst_independent_run/sky/corrected_avg_vis_1/"
    "sky_model.skymodel"
)
DEFAULT_OUTPUT_DIR = "./dp3_aligned_execute_output"

# Solver settings shared by DP3 execute and run_solver
NITER = 50
TOL = 1e-6


def clear_and_create_dir(path):
    """Delete ``path`` if it exists and create it empty."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def load_ms_weight_spectrum(ms_path, vis):
    """
    Read WEIGHT_SPECTRUM from the MS on the (time, baseline, freq, pol)
    grid of ``vis``.

    DP3 execute uses WEIGHT_SPECTRUM, whereas load_data_stage loads WEIGHT
    (all 1 in this MS), so run_solver needs these weights to match DP3.
    """
    with table(ms_path, ack=False) as ms:
        weight = ms.getcol("WEIGHT_SPECTRUM")
        ms_time = ms.getcol("TIME")
        ms_ant1, ms_ant2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")

    # Map each MS row to its (time, baseline) position in vis
    time_idx = np.searchsorted(vis.time.data, ms_time)
    bl_lookup = {
        (a1, a2): i
        for i, (a1, a2) in enumerate(zip(vis.antenna1.data, vis.antenna2.data))
    }
    bl_idx = np.array(
        [bl_lookup[(a1, a2)] for a1, a2 in zip(ms_ant1, ms_ant2)]
    )

    gridded = np.zeros(vis.weight.shape, dtype=np.float32)
    gridded[time_idx, bl_idx] = weight
    return gridded


def write_parset(content, path):
    """Write a DP3 parset and return its path."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    return path


def dp3_predict_model_column(ms_path, skymodel_path, output_dir):
    """Step 1: DP3 predict into MODEL_DATA of ``ms_path`` (in place)."""
    parset = f"""
msin = "{ms_path}"
msin.datacolumn = DATA
msout = .
msout.datacolumn = MODEL_DATA

steps = [predict]

predict.type = predict
predict.sourcedb = "{skymodel_path}"
"""
    dp3.execute(write_parset(parset, f"{output_dir}/predict.parset"), [])


def dp3_ddecal_execute(ms_path, h5parm_path, output_dir):
    """
    Step 2: DP3 DDECal via execute, reading the model from MODEL_DATA
    instead of predicting it from the sourcedb.
    """
    parset = f"""
msin = "{ms_path}"
msin.datacolumn = DATA
msout = ""

steps = [cal]

cal.type = ddecal
cal.modeldatacolumns = [MODEL_DATA]
cal.h5parm = "{h5parm_path}"
cal.mode = fulljones
cal.solint = 0
cal.nchan = 1
cal.maxiter = {NITER}
cal.tolerance = {TOL}
"""
    dp3.execute(write_parset(parset, f"{output_dir}/ddecal.parset"), [])


def inst_run_solver(ms_path, skymodel_path, output_dir, use_model_column):
    """
    Step 3: load DATA and MODEL_DATA into INST and run the dp3_ddecal
    solver. Returns the computed gaintable and the DATA visibility.
    """
    # The zarr cache key is "<ms>_fid<id>_ddid<id>" and does NOT include
    # the datacolumn, so each column needs its own cache_directory
    aligned_up_out = load_data_stage(
        UpstreamOutput(),
        output_dir,
        [ms_path],
        datacolumn="DATA",
        cache_directory=f"{output_dir}/cache_data",
    )[0]
    aligned_model_up_out = load_data_stage(
        UpstreamOutput(),
        output_dir,
        [ms_path],
        datacolumn="MODEL_DATA",
        cache_directory=f"{output_dir}/cache_model",
    )[0]

    aligned_vis = aligned_up_out.vis
    aligned_modelvis = aligned_model_up_out.vis

    # With skymodel_path, DDECal predicts the patch nearest the phase
    # centre itself and ignores aligned_modelvis. Without it, the
    # MODEL_DATA visibilities are passed to DDECal, as DP3 execute does.
    solver = Solver.get_solver(
        "dp3_ddecal",
        skymodel_path=None if use_model_column else skymodel_path,
        crosspol=True,
        niter=NITER,
        tol=TOL,
    )

    # DP3 execute uses WEIGHT_SPECTRUM, while load_data_stage loads WEIGHT
    # (all 1), so use the MS weights
    aligned_vis_ms_weights = aligned_vis.assign(
        weight=aligned_vis.weight.copy(
            data=load_ms_weight_spectrum(ms_path, aligned_vis)
        ).chunk(aligned_vis.weight.chunksizes)
    )

    aligned_solver_gaintable = run_solver(
        aligned_vis_ms_weights,
        aligned_modelvis,
        aligned_up_out.gaintable,
        solver,
    )
    computed = dask.compute(aligned_solver_gaintable, optimize_graph=True)[0]
    return computed, aligned_vis


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ms", default=DEFAULT_MS_PATH, help="Input MS")
    parser.add_argument(
        "--skymodel",
        default=DEFAULT_SKYMODEL_PATH,
        help="DP3 sky model used to predict MODEL_DATA",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (deleted and recreated on each run)",
    )
    parser.add_argument(
        "--use-model-column",
        action="store_true",
        help=(
            "Give MODEL_DATA to the INST solver instead of letting DDECal "
            "predict from --skymodel (the notebook's behaviour)"
        ),
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        help="Dask scheduler address, e.g. tcp://127.0.0.1:34567 "
        "(default: local dask scheduler)",
    )
    return parser.parse_args()


def main():
    """Run the aligned DP3 execute vs run_solver comparison."""
    args = parse_args()

    # Check inputs before deleting anything
    for path in (args.ms, args.skymodel):
        if not os.path.exists(path):
            sys.exit(f"Not found: {path}")

    if args.scheduler:
        # pylint: disable=import-outside-toplevel
        from distributed import Client

        Client(args.scheduler)

    output_dir = args.output_dir
    aligned_ms_path = f"{output_dir}/aligned.ms"
    aligned_dp3_gainpath = f"{output_dir}/gaincal_via_execute.h5parm"

    clear_and_create_dir(output_dir)
    # Work on a copy so that the input MS is never modified
    shutil.copytree(args.ms, aligned_ms_path)

    dp3_predict_model_column(aligned_ms_path, args.skymodel, output_dir)
    dp3_ddecal_execute(aligned_ms_path, aligned_dp3_gainpath, output_dir)

    solver_gaintable, aligned_vis = inst_run_solver(
        aligned_ms_path, args.skymodel, output_dir, args.use_model_column
    )

    dp3_gaintable = create_gaintable_from_h5parm(
        aligned_dp3_gainpath,
        interval=SolutionIntervals(aligned_vis.time.data, "full").intervals,
        vis=aligned_vis,
    )

    compare_arrays(
        solver_gaintable.CALPARAM_GAIN.data,
        dp3_gaintable.CALPARAM_GAIN.data,
        meta="Gain (solver vs DP3 execute)",
    )
    compare_arrays(
        solver_gaintable.CALPARAM_WEIGHT.data,
        dp3_gaintable.CALPARAM_WEIGHT.data,
        meta="Weight (solver vs DP3 execute)",
    )


if __name__ == "__main__":
    main()
