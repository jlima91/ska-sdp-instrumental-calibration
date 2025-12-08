from ska_sdp_instrumental_calibration import (
    target_calibration as target_pipeline,
)
from ska_sdp_instrumental_calibration.stages import (
    export_gaintable_stage,
    target_calibration,
)


def test_target_calibration_pipeline_definition():
    pipeline = target_pipeline.ska_sdp_instrumental_target_calibration
    stages = [
        target_calibration.load_data_stage,
        target_calibration.predict_vis_stage,
        target_calibration.complex_gain_calibration_stage,
        export_gaintable_stage,
    ]
    assert pipeline.name == "ska_sdp_instrumental_target_calibration"
    assert list(pipeline._stages) == stages


def test_target_ionospheric_calibration_pipeline_definition():
    pipeline = (
        target_pipeline.ska_sdp_instrumental_target_ionospheric_calibration
    )
    stages = [
        target_calibration.load_data_stage,
        target_calibration.predict_vis_stage,
        target_calibration.ionospheric_delay_stage,
        export_gaintable_stage,
    ]
    assert (
        pipeline.name == "ska_sdp_instrumental_target_ionospheric_calibration"
    )
    assert list(pipeline._stages) == stages
