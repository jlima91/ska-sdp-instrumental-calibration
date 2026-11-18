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
import logging
import os
import shutil
import sys
import time
from contextlib import contextmanager
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

logger = logging.getLogger("dp3_ddecal_aligned_comparison")

# Solver settings shared by DP3 execute and run_solver
NITER = 50
TOL = 1e-6


def setup_logging(level):
    """
    Log to stderr with timestamps. INST's own loggers (load_data_stage,
    chunking, ...) share the root handler, so their messages appear too.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextmanager
def log_step(name):
    """Log the start, end and duration of one step of the flow."""
    logger.info("=== START: %s", name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception(
            "=== FAILED: %s (after %.1f s)", name, time.perf_counter() - start
        )
        raise
    logger.info("=== DONE:  %s (%.1f s)", name, time.perf_counter() - start)


def log_ms_summary(ms_path):
    """Log the size and layout of an MS, to sanity-check the input."""
    with table(ms_path, ack=False) as ms:
        n_rows = ms.nrows()
        columns = ms.colnames()
        n_times = len(np.unique(ms.getcol("TIME")))
    with table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as spw:
        chan_freq = spw.getcol("CHAN_FREQ")
    with table(f"{ms_path}/ANTENNA", ack=False) as ant:
        n_ant = ant.nrows()

    logger.info(
        "MS %s: %d rows, %d times, %d antennas, %d channels "
        "(%.3f-%.3f MHz)",
        ms_path,
        n_rows,
        n_times,
        n_ant,
        chan_freq.shape[-1],
        chan_freq.min() / 1e6,
        chan_freq.max() / 1e6,
    )
    for col in ("DATA", "MODEL_DATA", "WEIGHT_SPECTRUM"):
        logger.info("  column %-15s present: %s", col, col in columns)


def log_skymodel_summary(skymodel_path):
    """Log how many sources and patches a DP3 text sky model holds."""
    n_sources = 0
    patches = set()
    with open(skymodel_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            # Skip blank lines, comments and the "format = ..." header
            if not line or line.startswith("#") or "format" in line.lower():
                continue
            fields = [f.strip() for f in line.split(",")]
            # Patch definition lines have an empty name: ", , patch, ..."
            if fields[0] == "" and len(fields) > 2:
                patches.add(fields[2])
            else:
                n_sources += 1
    logger.info(
        "Sky model %s: %d sources, %d patches",
        skymodel_path,
        n_sources,
        len(patches),
    )
    if n_sources > 10000:
        logger.warning(
            "Large sky model: DP3 predict may be slow; consider a higher "
            "flux limit"
        )


def clear_and_create_dir(path):
    """Delete ``path`` if it exists and create it empty."""
    if os.path.exists(path):
        logger.info("Deleting existing output directory %s", path)
        shutil.rmtree(path)
    os.mkdir(path)
    logger.info("Created output directory %s", os.path.abspath(path))


# def load_ms_weight_spectrum(ms_path, vis):
#     """
#     Read WEIGHT_SPECTRUM from the MS on the (time, baseline, freq, pol)
#     grid of ``vis``.

#     DP3 execute uses WEIGHT_SPECTRUM, whereas load_data_stage loads WEIGHT
#     (all 1 in this MS), so run_solver needs these weights to match DP3.
#     """
#     with table(ms_path, ack=False) as ms:
#         weight = ms.getcol("WEIGHT_SPECTRUM")
#         ms_time = ms.getcol("TIME")
#         ms_ant1, ms_ant2 = ms.getcol("ANTENNA1"), ms.getcol("ANTENNA2")

#     # Map each MS row to its (time, baseline) position in vis
#     time_idx = np.searchsorted(vis.time.data, ms_time)
#     bl_lookup = {
#         (a1, a2): i
#         for i, (a1, a2) in enumerate(zip(vis.antenna1.data, vis.antenna2.data))
#     }
#     bl_idx = np.array(
#         [bl_lookup[(a1, a2)] for a1, a2 in zip(ms_ant1, ms_ant2)]
#     )

#     gridded = np.zeros(vis.weight.shape, dtype=np.float32)
#     gridded[time_idx, bl_idx] = weight
#     logger.info(
#         "Gridded WEIGHT_SPECTRUM to %s (min %.3g, max %.3g)",
#         gridded.shape,
#         gridded.min(),
#         gridded.max(),
#     )
#     return gridded


def write_parset(content, path):
    """Write a DP3 parset and return its path."""
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
    logger.info("Wrote parset %s", path)
    # The full parset is useful when DP3 rejects a setting
    logger.debug("Parset contents:%s", content)
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
    logger.info("DP3 predict: %s -> MODEL_DATA of %s", skymodel_path, ms_path)
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
    logger.info(
        "DP3 DDECal: fulljones, solint=0, nchan=1, maxiter=%d, tol=%g",
        NITER,
        TOL,
    )
    dp3.execute(write_parset(parset, f"{output_dir}/ddecal.parset"), [])
    logger.info(
        "DP3 solutions written to %s (%.1f kB)",
        h5parm_path,
        os.path.getsize(h5parm_path) / 1024,
    )


def inst_run_solver(ms_path, skymodel_path, output_dir, use_model_column):
    """
    Step 3: load DATA and MODEL_DATA into INST and run the dp3_ddecal
    solver. Returns the computed gaintable and the DATA visibility.
    """
    # The zarr cache key is "<ms>_fid<id>_ddid<id>" and does NOT include
    # the datacolumn, so each column needs its own cache_directory
    logger.info(
        "Loading DATA column into INST (cache %s/cache_data)", output_dir
    )
    aligned_up_out = load_data_stage(
        UpstreamOutput(),
        output_dir,
        [ms_path],
        datacolumn="DATA",
        cache_directory=f"{output_dir}/cache_data",
    )[0]
    logger.info(
        "Loading MODEL_DATA column into INST (cache %s/cache_model)",
        output_dir,
    )
    aligned_model_up_out = load_data_stage(
        UpstreamOutput(),
        output_dir,
        [ms_path],
        datacolumn="MODEL_DATA",
        cache_directory=f"{output_dir}/cache_model",
    )[0]

    aligned_vis = aligned_up_out.vis
    aligned_modelvis = aligned_model_up_out.vis
    logger.info("vis dims: %s", dict(aligned_vis.sizes))
    logger.info("vis chunks: %s", dict(aligned_vis.chunksizes))
    logger.info("modelvis dims: %s", dict(aligned_modelvis.sizes))
    logger.info(
        "Initial gaintable dims: %s", dict(aligned_up_out.gaintable.sizes)
    )

    if use_model_column:
        logger.info("Solver model: MODEL_DATA visibilities (same as DP3)")
    else:
        logger.info(
            "Solver model: DDECal predicts from %s (patch nearest the "
            "phase centre); MODEL_DATA is ignored",
            skymodel_path,
        )

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
    # (all 1). To pass the MS weights instead, uncomment this and give
    # aligned_vis_ms_weights to run_solver below.
    # aligned_vis_ms_weights = aligned_vis.assign(
    #     weight=aligned_vis.weight.copy(
    #         data=load_ms_weight_spectrum(ms_path, aligned_vis)
    #     ).chunk(aligned_vis.weight.chunksizes)
    # )
    logger.info("Using the WEIGHT column loaded by load_data_stage")

    aligned_solver_gaintable = run_solver(
        aligned_vis,
        aligned_modelvis,
        aligned_up_out.gaintable,
        solver,
    )
    logger.info(
        "Built lazy dask graph (%d tasks); computing...",
        len(aligned_solver_gaintable.__dask_graph__()),
    )

    start = time.perf_counter()
    computed = dask.compute(aligned_solver_gaintable, optimize_graph=True)[0]
    logger.info(
        "run_solver computed in %.1f s; gaintable dims: %s",
        time.perf_counter() - start,
        dict(computed.sizes),
    )
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (DEBUG also prints the DP3 parsets)",
    )
    return parser.parse_args()


def main():
    """Run the aligned DP3 execute vs run_solver comparison."""
    args = parse_args()
    setup_logging(args.log_level)
    total_start = time.perf_counter()

    logger.info("DP3 DDECal aligned comparison starting")
    logger.info("  --ms               %s", args.ms)
    logger.info("  --skymodel         %s", args.skymodel)
    logger.info("  --output-dir       %s", args.output_dir)
    logger.info("  --use-model-column %s", args.use_model_column)
    logger.info("  --scheduler        %s", args.scheduler or "(local)")
    logger.info(
        "  OPENBLAS_NUM_THREADS=%s", os.environ.get("OPENBLAS_NUM_THREADS")
    )

    # Check inputs before deleting anything
    with log_step("Check inputs"):
        for path in (args.ms, args.skymodel):
            if not os.path.exists(path):
                logger.error("Not found: %s", path)
                sys.exit(f"Not found: {path}")
        # DP3 reads text sky models only with these suffixes; anything
        # else is treated as the unsupported binary sourcedb format
        if not args.skymodel.endswith((".skymodel", ".txt")):
            logger.warning(
                "Sky model does not end in .skymodel or .txt; DP3 will "
                "probably reject it as an old sourcedb"
            )
        log_ms_summary(args.ms)
        log_skymodel_summary(args.skymodel)

    if args.scheduler:
        # pylint: disable=import-outside-toplevel
        from distributed import Client

        with log_step(f"Connect to dask scheduler {args.scheduler}"):
            client = Client(args.scheduler)
            workers = client.scheduler_info()["workers"]
            logger.info(
                "Connected: %d workers, %d threads, dashboard %s",
                len(workers),
                sum(w["nthreads"] for w in workers.values()),
                client.dashboard_link,
            )
    else:
        logger.info("No --scheduler given: dask runs in this process")

    output_dir = args.output_dir
    aligned_ms_path = f"{output_dir}/aligned.ms"
    aligned_dp3_gainpath = f"{output_dir}/gaincal_via_execute.h5parm"

    with log_step("Prepare output directory and copy MS"):
        clear_and_create_dir(output_dir)
        # Work on a copy so that the input MS is never modified
        logger.info("Copying %s -> %s", args.ms, aligned_ms_path)
        shutil.copytree(args.ms, aligned_ms_path)

    with log_step("Step 1: DP3 predict into MODEL_DATA"):
        dp3_predict_model_column(aligned_ms_path, args.skymodel, output_dir)
        log_ms_summary(aligned_ms_path)

    with log_step("Step 2: DP3 DDECal execute"):
        dp3_ddecal_execute(aligned_ms_path, aligned_dp3_gainpath, output_dir)

    with log_step("Step 3: INST run_solver (dp3_ddecal)"):
        solver_gaintable, aligned_vis = inst_run_solver(
            aligned_ms_path, args.skymodel, output_dir, args.use_model_column
        )

    with log_step("Step 4: Load DP3 h5parm as a gaintable"):
        dp3_gaintable = create_gaintable_from_h5parm(
            aligned_dp3_gainpath,
            interval=SolutionIntervals(
                aligned_vis.time.data, "full"
            ).intervals,
            vis=aligned_vis,
        )
        # The schema class is an xarray.Dataset at runtime
        logger.info(
            "DP3 gaintable dims: %s",
            dict(dp3_gaintable.sizes),  # pylint: disable=no-member
        )

    # compare_arrays prints its own report below each of these lines
    logger.info("=== Step 5: Compare INST run_solver against DP3 execute")
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

    logger.info(
        "Comparison finished in %.1f s; outputs in %s",
        time.perf_counter() - total_start,
        os.path.abspath(output_dir),
    )


if __name__ == "__main__":
    main()
