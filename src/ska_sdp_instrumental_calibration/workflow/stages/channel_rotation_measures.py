import os
from copy import deepcopy

import dask
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ...data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    predict_vis,
    run_solver,
)
from ...data_managers.data_export import export_gaintable_to_h5parm
from ...processing_tasks.rotation_measures import model_rotations
from ..utils import plot_gaintable, plot_rm_station
from ._common import RUN_SOLVER_DOCSTRING, RUN_SOLVER_NESTED_CONFIG


@ConfigurableStage(
    "generate_channel_rm",
    configuration=Configuration(
        fchunk=ConfigParam(
            int,
            -1,
            description="""Number of frequency channels per chunk.
            If set to -1, use fchunk value from load_data""",
        ),
        peak_threshold=ConfigParam(
            float,
            0.5,
            description="""Height of peak in the RM spectrum required
            for a rotation detection.""",
        ),
        refine_fit=ConfigParam(
            bool,
            True,
            description="""Whether or not to refine the RM spectrum
            peak locations with a nonlinear optimisation of
            the station RM values.""",
        ),
        plot_rm=ConfigParam(
            bool,
            False,
            description="""Plot the estimated rotational measures
            per station""",
        ),
        plot_table=ConfigParam(
            bool, False, description="Plot the generated gain table"
        ),
        run_solver_config=deepcopy(RUN_SOLVER_NESTED_CONFIG),
        export_gaintable=ConfigParam(
            bool,
            False,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def generate_channel_rm_stage(
    upstream_output,
    fchunk,
    peak_threshold,
    refine_fit,
    plot_rm,
    plot_table,
    run_solver_config,
    export_gaintable,
    _output_dir_,
):
    """
    Generates channel rotation measures

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        fchunk: int
            Number of frequency channels per chunk.
            If it is '-1' fchunk of load_data will be used.
        peak_threshold: float
            Height of peak in the RM spectrum required
            for a rotation detection.
        refine_fit: bool
            Whether or not to refine the RM spectrum peak
            locations with a nonlinear optimisation
            of the station RM values.
        plot_rm: bool
            Plot the estimated rotational measures
            per station.
        plot_table: bool
            Plot the gaintable.
        run_solver_config: dict
            {run_solver_docstring}
        export_gaintable: bool
            Export intermediate gain solutions
        _output_dir_ : str
            Directory path where the output file will be written.
            Provided by piper.

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    vis = upstream_output.vis
    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable
    if fchunk != -1:
        initialtable = upstream_output.gaintable.chunk({"frequency": fchunk})

    call_counter_suffix = ""
    if call_count := upstream_output.get_call_count("channel_rm"):
        call_counter_suffix = f"_{call_count}"

    path_prefix = os.path.join(
        _output_dir_, f"channel_rm{call_counter_suffix}"
    )

    rotations = model_rotations(
        initialtable,
        peak_threshold=peak_threshold,
        refine_fit=refine_fit,
        refant=run_solver_config["refant"],
    )
    modelvis = predict_vis(
        vis,
        upstream_output["lsm"],
        beam_type=upstream_output["beam_type"],
        eb_ms=upstream_output["eb_ms"],
        eb_coeffs=upstream_output["eb_coeffs"],
        station_rm=rotations.rm_est,
    )
    if upstream_output["beams"] is not None:
        modelvis = apply_gaintable_to_dataset(
            modelvis, upstream_output["beams"], inverse=True
        )

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        **run_solver_config,
    )

    if plot_rm:
        upstream_output.add_compute_tasks(
            plot_rm_station(
                initialtable,
                **rotations.get_plot_params_for_station(),
                plot_path_prefix=path_prefix,
            )
        )

    if plot_table:
        upstream_output.add_compute_tasks(
            plot_gaintable(
                gaintable,
                path_prefix,
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            )
        )

    if export_gaintable:
        gaintable_file_path = os.path.join(
            _output_dir_, f"channel_rm{call_counter_suffix}.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("channel_rm")

    return upstream_output


generate_channel_rm_stage.__doc__ = generate_channel_rm_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
