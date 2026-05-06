import logging
from typing import Annotated

import dask
from pydantic import Field
from ska_sdp_piper.piper import ConfigurableStage

from ..data_managers.data_export import export_gaintable_to_h5parm
from ..numpy_processors.solvers import Solver
from ..xarray_processors import parse_antenna
from ..xarray_processors.solver import run_solver
from ._utils import get_gaintables_path

logger = logging.getLogger()


@ConfigurableStage(name="bandpass_initialisation")
def bandpass_initialisation_stage(
    _upstream_output_,
    _qa_dir_,
    refant: Annotated[int | str, Field(description="Reference antenna")] = 0,
    niter: Annotated[
        int, Field(description="Number of solver iterations.")
    ] = 200,
    tol: Annotated[
        float,
        Field(
            description="""Iteration stops when the fractional change
                in the gain solution is below this tolerance."""
        ),
    ] = 1e-06,
    export_gaintable: Annotated[
        bool, Field(description="Export intermediate gain solutions.")
    ] = True,
):
    """
    Initialises the gains for bandpass calibration

    Parameters
    ----------
        _upstream_output_: dict
            Output from the upstream stage
        _qa_dir_: str
            Directory path where the diagnostic QA outputs will be written.
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

    _upstream_output_.add_checkpoint_key("gaintable")
    vis = _upstream_output_.vis
    modelvis = _upstream_output_.modelvis
    initialtable = _upstream_output_.gaintable
    prefix = _upstream_output_.ms_prefix

    refant = parse_antenna(refant, initialtable.configuration.names)
    solver = Solver.get_solver(refant=refant, niter=niter, tol=tol)

    gaintable = run_solver(
        vis=vis,
        modelvis=modelvis,
        gaintable=initialtable,
        solver=solver,
    )

    if export_gaintable:
        gaintable_file_path = get_gaintables_path(
            _qa_dir_, f"{prefix}/bandpass_initialisation.gaintable.h5parm"
        )

        _upstream_output_.add_compute_tasks(
            dask.delayed(export_gaintable_to_h5parm)(
                gaintable, gaintable_file_path
            )
        )

    _upstream_output_["gaintable"] = gaintable
    _upstream_output_["refant"] = refant

    return _upstream_output_
