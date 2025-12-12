from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages import (
    bandpass_initialisation_stage,
)


def test_should_have_the_expected_default_configuration():
    expected_config = {
        "bandpass_initialisation": {
            "refant": 0,
            "niter": 200,
            "tol": 1.0e-06,
            "export_gaintable": True,
        }
    }

    assert bandpass_initialisation_stage.config == expected_config


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".parse_reference_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".SolverFactory"
)
def test_should_initialize_gains_for_bandpass(
    solver_factory_mock, run_solver_mock, parse_ref_ant_mock
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_gaintable = "initial_gaintable"
    upstream_output["gaintable"] = initial_gaintable
    parse_ref_ant_mock.return_value = 0
    tol = 1e-06
    refant = 0
    niter = 200
    solver_factory_mock.get_solver.return_value = "SOLVER"

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock

    actual = bandpass_initialisation_stage.stage_definition(
        upstream_output,
        refant=refant,
        niter=niter,
        tol=tol,
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    parse_ref_ant_mock.assert_called_once_with(0, initial_gaintable)

    solver_factory_mock.get_solver.assert_called_once_with(
        refant=refant,
        niter=niter,
        tol=tol,
    )
    run_solver_mock.assert_called_once_with(
        vis=upstream_output.vis,
        modelvis=upstream_output.modelvis,
        gaintable=initial_gaintable,
        solver="SOLVER",
    )

    assert actual.gaintable == gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_calibration"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".parse_reference_antenna"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".SolverFactory"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.stages.bandpass_initialisation"
    ".get_gaintables_path"
)
def test_should_export_gaintable(
    get_gaintables_path_mock,
    export_gaintable_mock,
    solver_factory_mock,
    run_solver_mock,
    parse_ref_ant_mock,
    delayed_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    initial_gaintable = "initial_gaintable"
    upstream_output["gaintable"] = initial_gaintable
    parse_ref_ant_mock.return_value = 0
    tol = 1e-06
    refant = 0
    niter = 200
    solver_factory_mock.get_solver.return_value = "SOLVER"

    gaintable_mock = Mock(name="gaintable")
    run_solver_mock.return_value = gaintable_mock
    get_gaintables_path_mock.return_value = (
        "/output/path/gaintables/bandpass_initialisation.gaintable.h5parm"
    )

    actual = bandpass_initialisation_stage.stage_definition(
        upstream_output,
        refant=refant,
        niter=niter,
        tol=tol,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    get_gaintables_path_mock.assert_called_once_with(
        "/output/path", "bandpass_initialisation.gaintable.h5parm"
    )

    export_gaintable_mock.assert_called_once_with(
        gaintable_mock,
        "/output/path/gaintables/bandpass_initialisation.gaintable.h5parm",
    )

    delayed_mock.assert_called_once_with(export_gaintable_mock)

    assert actual.gaintable == gaintable_mock
