import logging
import os
from copy import deepcopy

import dask
from ska_sdp_piper.piper.configurations import (
    ConfigParam,
    Configuration,
    NestedConfigParam,
)
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.processing_tasks.predict_model.predict import (  # noqa: E501
    predict_vis,
)

from ...data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
    run_solver,
)
from ...data_managers.data_export import export_gaintable_to_h5parm
from ...processing_tasks.rotation_measures import model_rotations
from ..utils import plot_bandpass_stages, plot_gaintable, plot_rm_station
from ._common import RUN_SOLVER_DOCSTRING, RUN_SOLVER_NESTED_CONFIG

logger = logging.getLogger()


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
        visibility_key=ConfigParam(
            str,
            "vis",
            description="Visibility data to be used for calibration.",
            allowed_values=["vis", "corrected_vis"],
        ),
        plot_rm_config=NestedConfigParam(
            "Plot Parameters for rotational measures",
            plot_rm=ConfigParam(
                bool,
                False,
                description="""Plot the estimated rotational measures
                per station""",
            ),
            station=ConfigParam(
                int,
                0,
                description="""Station number to be plotted""",
                nullable=True,
            ),
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
    visibility_key,
    plot_rm_config,
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
        visibility_key: str
            Visibility data to be used for calibration.
        plot_rm_config:
            Configs required for RM plots.
            eg: {{plot_rm: False, station: 0}}
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

    vis = upstream_output[visibility_key]
    logger.info(f"Using {visibility_key} for calibration.")

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

    modelvis_xda = predict_vis(
        vis.vis,
        vis.uvw,
        vis.datetime,
        vis.configuration,
        vis.antenna1,
        vis.antenna2,
        upstream_output["lsm"],
        vis.phasecentre,
        beam_type=upstream_output["beam_type"],
        eb_ms=upstream_output["eb_ms"],
        eb_coeffs=upstream_output["eb_coeffs"],
        station_rm=rotations.rm_est,
    )

    modelvis = vis.assign({"vis": modelvis_xda})

    if upstream_output["beams"] is not None:
        modelvis = apply_gaintable_to_dataset(
            modelvis, upstream_output["beams"], inverse=True
        )

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        **run_solver_config,
    )

    if plot_rm_config["plot_rm"]:
        upstream_output.add_compute_tasks(
            plot_bandpass_stages(
                gaintable,
                initialtable,
                rotations.rm_est,
                run_solver_config["refant"],
                plot_path_prefix=path_prefix,
            ),
            plot_rm_station(
                initialtable,
                **rotations.get_plot_params_for_station(
                    plot_rm_config["station"]
                ),
                plot_path_prefix=path_prefix,
            ),
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

    upstream_output["modelvis"] = modelvis
    upstream_output["gaintable"] = gaintable
    upstream_output.increment_call_count("channel_rm")

    return upstream_output


generate_channel_rm_stage.__doc__ = generate_channel_rm_stage.__doc__.format(
    run_solver_docstring=RUN_SOLVER_DOCSTRING
)
