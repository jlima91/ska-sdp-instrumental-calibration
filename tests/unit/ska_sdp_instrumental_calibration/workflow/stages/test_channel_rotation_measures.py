from mock import MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.workflow.stages import (
    generate_channel_rm_stage,
)


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.predict_vis"
)
def test_should_gen_channel_rm_using_predict_model_vis_when_beam_is_none(
    predict_vis_mock,
    apply_gaintable_mock,
    model_rotations_mock,
    run_solver_mock,
    delayed_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["lsm"] = Mock(name="lsm")
    upstream_output["eb_ms"] = Mock(name="eb_ms")
    upstream_output["eb_coeffs"] = Mock(name="eb_coeffs")
    upstream_output["refant"] = Mock(name="refant")
    upstream_output["beam_type"] = Mock(name="beam_type")
    upstream_output["beams"] = None
    new_model_vis_mock = Mock(name="new model vis")
    upstream_output["corrected_vis"].assign.return_value = new_model_vis_mock
    initial_table_mock = MagicMock(name="initial gaintable")
    solved_gaintable_mock = Mock(name="run solver gaintable")
    initial_table_mock.chunk.return_value = Mock(
        name="initial gaintable chunk"
    )
    model_rotations_obj_mock = MagicMock(name="model rotation mock")
    rm_est_mock = Mock(name="rm est")
    model_rotations_obj_mock.rm_est = rm_est_mock
    model_rotations_mock.return_value = model_rotations_obj_mock

    upstream_output["gaintable"] = initial_table_mock
    predict_vis_mock.return_value = new_model_vis_mock
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_rm_config = {
        "plot_rm": False,
        "station": 1,
    }
    result = generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=-1,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=False,
        export_gaintable=False,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    model_rotations_mock.assert_called_once_with(
        initial_table_mock,
        peak_threshold=0.5,
        refine_fit=False,
        refant=run_solver_config["refant"],
    )

    predict_vis_mock.assert_called_once_with(
        upstream_output["corrected_vis"].vis,
        upstream_output["corrected_vis"].uvw,
        upstream_output["corrected_vis"].datetime,
        upstream_output["corrected_vis"].configuration,
        upstream_output["corrected_vis"].antenna1,
        upstream_output["corrected_vis"].antenna2,
        upstream_output["lsm"],
        upstream_output["corrected_vis"].phasecentre,
        beam_type=upstream_output["beam_type"],
        eb_ms=upstream_output["eb_ms"],
        eb_coeffs=upstream_output["eb_coeffs"],
        station_rm=rm_est_mock,
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output["corrected_vis"],
        modelvis=new_model_vis_mock,
        solver="solver",
        niter=1,
        refant=2,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        jones_type="T",
        timeslice=None,
    )

    assert result["modelvis"] == new_model_vis_mock
    assert result["gaintable"] == solved_gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.predict_vis"
)
def test_should_apply_beam_to_model_vis_when_beam_is_not_none(
    predict_vis_mock,
    apply_gaintable_mock,
    model_rotations_mock,
    run_solver_mock,
    delayed_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["lsm"] = Mock(name="lsm")
    upstream_output["eb_ms"] = Mock(name="eb_ms")
    upstream_output["eb_coeffs"] = Mock(name="eb_coeffs")
    upstream_output["refant"] = Mock(name="refant")
    upstream_output["beam_type"] = Mock(name="beam_type")
    upstream_output["beams"] = Mock(name="beams")
    initial_table_mock = Mock(name="initial gaintable")
    initial_table_mock.chunk.return_value = Mock(
        name="initial gaintable chunk"
    )
    upstream_output["gaintable"] = initial_table_mock
    new_model_vis_mock = Mock(name="new model vis")
    beam_model_vis = Mock(name="beam applied model vis")

    predict_vis_mock.return_value = new_model_vis_mock

    model_rotations_obj_mock = MagicMock(name="model rotation mock")
    upstream_output["corrected_vis"].assign.return_value = new_model_vis_mock
    rm_est_mock = Mock(name="rm est")
    model_rotations_obj_mock.rm_est = rm_est_mock
    model_rotations_mock.return_value = model_rotations_obj_mock

    apply_gaintable_mock.return_value = beam_model_vis
    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_rm_config = {
        "plot_rm": False,
        "station": 1,
    }
    result = generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=-1,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=False,
        export_gaintable=False,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    model_rotations_mock.assert_called_once_with(
        initial_table_mock,
        peak_threshold=0.5,
        refine_fit=False,
        refant=run_solver_config["refant"],
    )

    apply_gaintable_mock.assert_called_once_with(
        new_model_vis_mock, upstream_output["beams"], inverse=True
    )

    predict_vis_mock.assert_called_once_with(
        upstream_output["corrected_vis"].vis,
        upstream_output["corrected_vis"].uvw,
        upstream_output["corrected_vis"].datetime,
        upstream_output["corrected_vis"].configuration,
        upstream_output["corrected_vis"].antenna1,
        upstream_output["corrected_vis"].antenna2,
        upstream_output["lsm"],
        upstream_output["corrected_vis"].phasecentre,
        beam_type=upstream_output["beam_type"],
        eb_ms=upstream_output["eb_ms"],
        eb_coeffs=upstream_output["eb_coeffs"],
        station_rm=rm_est_mock,
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output["corrected_vis"],
        modelvis=beam_model_vis,
        solver="solver",
        niter=1,
        refant=2,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        jones_type="T",
        timeslice=None,
    )
    assert result["modelvis"] == beam_model_vis
    assert result["gaintable"] == solved_gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.predict_vis"
)
def test_should_generate_channel_rm_using_provided_fchunk(
    predict_vis_mock,
    model_rotations_mock,
    run_solver_mock,
    delayed_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["lsm"] = Mock(name="lsm")
    upstream_output["eb_ms"] = Mock(name="eb_ms")
    upstream_output["eb_coeffs"] = Mock(name="eb_coeffs")
    upstream_output["refant"] = Mock(name="refant")
    upstream_output["beam_type"] = Mock(name="beam_type")
    upstream_output["beams"] = None
    initial_table_mock = Mock(name="initial gaintable")
    new_model_vis_mock = Mock(name="new model vis")
    chunked_table_mock = Mock(name="chunked gaintable")
    solved_gaintable_mock = Mock(name="run solver gaintable")

    upstream_output["gaintable"] = initial_table_mock
    initial_table_mock.chunk.return_value = chunked_table_mock
    model_rotations_obj_mock = MagicMock(name="model rotation mock")
    upstream_output["corrected_vis"].assign.return_value = new_model_vis_mock
    rm_est_mock = Mock(name="rm est")
    model_rotations_obj_mock.rm_est = rm_est_mock
    model_rotations_mock.return_value = model_rotations_obj_mock

    predict_vis_mock.return_value = new_model_vis_mock
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_rm_config = {
        "plot_rm": False,
        "station": 1,
    }
    result = generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=False,
        run_solver_config=run_solver_config,
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    initial_table_mock.chunk.assert_called_once_with({"frequency": 40})
    model_rotations_mock.assert_called_once_with(
        chunked_table_mock,
        peak_threshold=0.5,
        refine_fit=False,
        refant=run_solver_config["refant"],
    )

    predict_vis_mock.assert_called_once_with(
        upstream_output["corrected_vis"].vis,
        upstream_output["corrected_vis"].uvw,
        upstream_output["corrected_vis"].datetime,
        upstream_output["corrected_vis"].configuration,
        upstream_output["corrected_vis"].antenna1,
        upstream_output["corrected_vis"].antenna2,
        upstream_output["lsm"],
        upstream_output["corrected_vis"].phasecentre,
        beam_type=upstream_output["beam_type"],
        eb_ms=upstream_output["eb_ms"],
        eb_coeffs=upstream_output["eb_coeffs"],
        station_rm=rm_est_mock,
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output.corrected_vis,
        modelvis=new_model_vis_mock,
        solver="solver",
        niter=1,
        refant=2,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        jones_type="T",
        timeslice=None,
    )

    assert result["gaintable"] == solved_gaintable_mock


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".plot_rm_station"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".plot_bandpass_stages"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".plot_gaintable"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.predict_vis"
)
def test_should_plot_with_proper_suffix(
    predict_vis_mock,
    model_rotations_mock,
    run_solver_mock,
    export_h5parm_mock,
    plot_gaintable_mock,
    plot_bandpass_stages_mock,
    plot_rm_station_mock,
    delayed_mock,
):

    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["lsm"] = Mock(name="lsm")
    upstream_output["eb_ms"] = Mock(name="eb_ms")
    upstream_output["eb_coeffs"] = Mock(name="eb_coeffs")
    upstream_output["refant"] = Mock(name="refant")
    upstream_output["beam_type"] = Mock(name="beam_type")
    upstream_output["beams"] = None
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    chunked_table_mock = Mock(name="chunked gaintable")
    initial_table_mock.chunk.return_value = chunked_table_mock

    model_rotations_obj_mock = MagicMock(name="model rotation mock")

    rm_est_mock = Mock(name="rm est")
    model_rotations_obj_mock.rm_est = rm_est_mock
    model_rotations_obj_mock.get_plot_params_for_station = Mock(
        name="get_plot_params_for_station", return_value={"rm_vals": "rm_vals"}
    )
    model_rotations_mock.return_value = model_rotations_obj_mock

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_rm_config = {
        "plot_rm": True,
        "station": 1,
    }
    generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=True,
        run_solver_config=run_solver_config,
        export_gaintable=False,
        _output_dir_="/output/path",
    )

    upstream_output["gaintable"] = initial_table_mock
    generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=True,
        run_solver_config=run_solver_config,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    model_rotations_mock.assert_has_calls(
        [
            call(
                chunked_table_mock,
                peak_threshold=0.5,
                refine_fit=False,
                refant=2,
            ),
            call(
                chunked_table_mock,
                peak_threshold=0.5,
                refine_fit=False,
                refant=2,
            ),
        ]
    )

    plot_bandpass_stages_mock.assert_has_calls(
        [
            call(
                solved_gaintable_mock,
                chunked_table_mock,
                rm_est_mock,
                2,
                plot_path_prefix="/output/path/channel_rm",
            ),
            call(
                solved_gaintable_mock,
                chunked_table_mock,
                rm_est_mock,
                2,
                plot_path_prefix="/output/path/channel_rm_1",
            ),
        ]
    )
    plot_rm_station_mock.assert_has_calls(
        [
            call(
                chunked_table_mock,
                rm_vals="rm_vals",
                plot_path_prefix="/output/path/channel_rm",
            ),
            call(
                chunked_table_mock,
                rm_vals="rm_vals",
                plot_path_prefix="/output/path/channel_rm_1",
            ),
        ]
    )

    plot_gaintable_mock.assert_has_calls(
        [
            call(
                solved_gaintable_mock,
                "/output/path/channel_rm",
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            ),
            call(
                solved_gaintable_mock,
                "/output/path/channel_rm_1",
                figure_title="Channel Rotation Measure",
                drop_cross_pols=True,
            ),
        ]
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".export_gaintable_to_h5parm"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures"
    ".model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.predict_vis"
)
def test_should_export_gaintable_with_proper_suffix(
    predict_vis_mock,
    model_rotations_mock,
    run_solver_mock,
    export_gaintable_mock,
    delayed_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["lsm"] = Mock(name="lsm")
    upstream_output["eb_ms"] = Mock(name="eb_ms")
    upstream_output["eb_coeffs"] = Mock(name="eb_coeffs")
    upstream_output["refant"] = Mock(name="refant")
    upstream_output["beam_type"] = Mock(name="beam_type")
    upstream_output["beams"] = None
    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    initial_table_mock = Mock(name="initial gaintable")
    upstream_output["gaintable"] = initial_table_mock

    chunked_table_mock = Mock(name="chunked gaintable")
    initial_table_mock.chunk.return_value = chunked_table_mock

    model_rotations_obj_mock = MagicMock(name="model rotation mock")
    rm_est_mock = Mock(name="rm est")
    model_rotations_obj_mock.rm_est = rm_est_mock
    model_rotations_mock.return_value = model_rotations_obj_mock

    solved_gaintable_mock = Mock(name="run solver gaintable")
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_rm_config = {
        "plot_rm": False,
        "station": 1,
    }
    generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=True,
        run_solver_config=run_solver_config,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    upstream_output["gaintable"] = initial_table_mock
    generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=40,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="corrected_vis",
        plot_rm_config=plot_rm_config,
        plot_table=True,
        run_solver_config=run_solver_config,
        export_gaintable=True,
        _output_dir_="/output/path",
    )

    export_gaintable_mock.assert_has_calls(
        [
            call(
                solved_gaintable_mock,
                "/output/path/channel_rm.gaintable.h5parm",
            ),
            call(
                solved_gaintable_mock,
                "/output/path/channel_rm_1.gaintable.h5parm",
            ),
        ]
    )

    delayed_mock.assert_has_calls(
        [call(export_gaintable_mock), call(export_gaintable_mock)]
    )


@patch(
    "ska_sdp_instrumental_calibration.workflow.stages"
    ".channel_rotation_measures"
    ".dask.delayed",
    side_effect=lambda x: x,
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.run_solver"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.model_rotations"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.apply_gaintable_to_dataset"
)
@patch(
    "ska_sdp_instrumental_calibration.workflow.stages."
    "channel_rotation_measures.predict_vis"
)
def test_should_not_use_corrected_vis_in_run_solver_when_config_is_false(
    predict_vis_mock,
    apply_gaintable_mock,
    model_rotations_mock,
    run_solver_mock,
    delayed_mock,
):
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["corrected_vis"] = Mock(name="corrected_vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["lsm"] = Mock(name="lsm")
    upstream_output["eb_ms"] = Mock(name="eb_ms")
    upstream_output["eb_coeffs"] = Mock(name="eb_coeffs")
    upstream_output["refant"] = Mock(name="refant")
    upstream_output["beam_type"] = Mock(name="beam_type")
    upstream_output["beams"] = None
    new_model_vis_mock = Mock(name="new model vis")
    upstream_output["vis"].assign.return_value = new_model_vis_mock
    initial_table_mock = MagicMock(name="initial gaintable")
    solved_gaintable_mock = Mock(name="run solver gaintable")
    initial_table_mock.chunk.return_value = Mock(
        name="initial gaintable chunk"
    )
    model_rotations_obj_mock = MagicMock(name="model rotation mock")
    rm_est_mock = Mock(name="rm est")
    model_rotations_obj_mock.rm_est = rm_est_mock
    model_rotations_mock.return_value = model_rotations_obj_mock

    upstream_output["gaintable"] = initial_table_mock
    predict_vis_mock.return_value = new_model_vis_mock
    run_solver_mock.return_value = solved_gaintable_mock

    run_solver_config = {
        "solver": "solver",
        "niter": 1,
        "refant": 2,
        "phase_only": False,
        "tol": 1e-06,
        "crosspol": False,
        "normalise_gains": "mean",
        "jones_type": "T",
        "timeslice": None,
    }
    plot_rm_config = {
        "plot_rm": False,
        "station": 1,
    }
    result = generate_channel_rm_stage.stage_definition(
        upstream_output,
        fchunk=-1,
        peak_threshold=0.5,
        refine_fit=False,
        visibility_key="vis",
        plot_rm_config=plot_rm_config,
        plot_table=False,
        export_gaintable=False,
        run_solver_config=run_solver_config,
        _output_dir_="/output/path",
    )

    model_rotations_mock.assert_called_once_with(
        initial_table_mock,
        peak_threshold=0.5,
        refine_fit=False,
        refant=run_solver_config["refant"],
    )

    predict_vis_mock.assert_called_once_with(
        upstream_output["vis"].vis,
        upstream_output["vis"].uvw,
        upstream_output["vis"].datetime,
        upstream_output["vis"].configuration,
        upstream_output["vis"].antenna1,
        upstream_output["vis"].antenna2,
        upstream_output["lsm"],
        upstream_output["vis"].phasecentre,
        beam_type=upstream_output["beam_type"],
        eb_ms=upstream_output["eb_ms"],
        eb_coeffs=upstream_output["eb_coeffs"],
        station_rm=rm_est_mock,
    )

    run_solver_mock.assert_called_once_with(
        vis=upstream_output["vis"],
        modelvis=new_model_vis_mock,
        solver="solver",
        niter=1,
        refant=2,
        phase_only=False,
        tol=1e-06,
        crosspol=False,
        normalise_gains="mean",
        jones_type="T",
        timeslice=None,
    )

    assert result["modelvis"] == new_model_vis_mock
    assert result["gaintable"] == solved_gaintable_mock
