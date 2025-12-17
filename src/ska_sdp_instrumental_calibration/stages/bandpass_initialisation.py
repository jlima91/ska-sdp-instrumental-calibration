import logging

import dask
from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..numpy_processors.solvers import SolverFactory
from ..xarray_processors import parse_antenna
from ..xarray_processors.solver import run_solver
from ._utils import get_gaintables_path

logger = logging.getLogger()


@ConfigurableStage(
    "bandpass_initialisation",
    configuration=Configuration(
        refant=ConfigParam(
            (int, str),
            0,
            description="""Reference antenna""",
            nullable=False,
        ),
        niter=ConfigParam(
            int,
            200,
            description="""Number of solver iterations.""",
            nullable=False,
        ),
        tol=ConfigParam(
            float,
            1e-06,
            description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance.""",
            nullable=False,
        ),
        export_gaintable=ConfigParam(
            bool,
            True,
            description="Export intermediate gain solutions.",
            nullable=False,
        ),
    ),
)
def bandpass_initialisation_stage(
    upstream_output,
    refant,
    niter,
    tol,
    export_gaintable,
    _output_dir_,
):
    """
    Initialises the gains for bandpass calibration

    Parameters
    ----------
        upstream_output: dict
            Output from the upstream stage
        refant: (int,str)
            Reference antenna
        niter: int
            Number of solver iterations
        tol: float
            Tolerance value for gain solution
        export_gaintable: bool
            Export intermediate gain solutions

    Returns
    -------
        dict
            Updated upstream_output with gaintable
    """

    upstream_output.add_checkpoint_key("gaintable")
    vis = upstream_output.vis
    modelvis = upstream_output.modelvis
    initialtable = upstream_output.gaintable

    refant = parse_antenna(
        refant, initialtable.configuration.names, initialtable.antenna1.size
    )
    solver = SolverFactory.get_solver(refant=refant, niter=niter, tol=tol)

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initialtable,
        solver=solver,
    )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _output_dir_, "bandpass_initialisation.gaintable.h5parm"
        )

        upstream_output.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    upstream_output["gaintable"] = gaintable

    return upstream_output
