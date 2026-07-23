import pytest
from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.scheduler import (
    InstrumentalDaskRunner,
    UpstreamOutput,
)


class TestUpstreamOutput:
    def test_should_return_manage_stage_count(self):
        upstream_output = UpstreamOutput()

        assert upstream_output.get_call_count("stage1") == 0
        upstream_output.increment_call_count("stage1")
        assert upstream_output.get_call_count("stage1") == 1
        upstream_output.increment_call_count("stage1")
        assert upstream_output.get_call_count("stage1") == 2

        assert upstream_output.get_call_count("stage2") == 0
        upstream_output.increment_call_count("stage2")
        assert upstream_output.get_call_count("stage2") == 1
        upstream_output.increment_call_count("stage2")
        assert upstream_output.get_call_count("stage2") == 2

    def test_set_and_get_item(self):
        upstream_output = UpstreamOutput()
        upstream_output["foo"] = 123
        assert upstream_output["foo"] == 123

    def test_getitem_raises_attribute_error(self):
        upstream_output = UpstreamOutput()
        with pytest.raises(AttributeError):
            _ = upstream_output["missing"]

    def test_getattr_returns_value(self):
        upstream_output = UpstreamOutput()
        upstream_output["bar"] = "baz"
        assert upstream_output.bar == "baz"

    def test_getattr_raises_attribute_error(self):
        upstream_output = UpstreamOutput()
        with pytest.raises(AttributeError):
            _ = upstream_output.missing

    def test_contains(self):
        upstream_output = UpstreamOutput()
        upstream_output["x"] = 1
        assert "x" in upstream_output
        assert "y" not in upstream_output

    @patch(
        "ska_sdp_instrumental_calibration.scheduler.scheduler"
        ".multiply_gaintables"
    )
    def test_build_calibration_table(self, multiply_mock):
        upstream_output = UpstreamOutput()
        delay_gaintable = MagicMock(name="delay")
        delay_table_copy = MagicMock(name="delay_copy")
        delay_gaintable.copy.return_value = delay_table_copy
        gaintable = MagicMock(name="gaintable")
        flux_gaintable = MagicMock(name="flux")

        upstream_output["gaintable"] = gaintable

        assert upstream_output.calibration_table == gaintable

        upstream_output["delay"] = delay_gaintable
        upstream_output["flux"] = flux_gaintable

        upstream_output.add_calibration_table("delay")
        upstream_output.add_calibration_table("flux")
        upstream_output.add_calibration_table("gaintable")

        assert upstream_output.calibration_table == multiply_mock.return_value
        multiply_mock.assert_has_calls(
            [
                call(delay_table_copy, flux_gaintable),
                call(multiply_mock.return_value, gaintable),
            ]
        )


class TestInstrumentalDaskRunner:
    @patch("ska_sdp_instrumental_calibration.scheduler.scheduler.task_manager")
    def test_should_compute_tasks_on_execute(self, mock_task_manager):
        pipeline = MagicMock(name="pipeline")

        dummy_stage = MagicMock()
        dummy_stage.name = "stage1"
        pipeline.executable_stages = [dummy_stage]

        scheduler = InstrumentalDaskRunner(_pipeline_=pipeline)
        scheduler.execute()

        # Check persist called for both stages
        mock_task_manager.compute.assert_called_once()

    @pytest.mark.skip(reason="Need to implement")
    def test_should_broadcast_and_aggregate():
        pass
