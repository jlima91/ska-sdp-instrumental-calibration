import pytest
from mock import MagicMock, patch

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

    def test_add_compute_tasks_and_property(self):
        upstream_output = UpstreamOutput()
        task1 = MagicMock()
        task2 = MagicMock()
        upstream_output.add_compute_tasks(task1, task2)
        assert task1 in upstream_output.compute_tasks
        assert task2 in upstream_output.compute_tasks

    def test_add_checkpoint_key(self):
        upstream_output = UpstreamOutput()
        upstream_output.add_checkpoint_key("key1", "key2")
        assert "key1" in upstream_output.checkpoint_keys
        assert "key2" in upstream_output.checkpoint_keys


class TestInstrumentalDaskRunner:
    @patch(
        "ska_sdp_instrumental_calibration.scheduler.get_client",
        side_effect=Exception("No client"),
    )
    @patch("ska_sdp_instrumental_calibration.scheduler.dask.persist")
    def test_should_persist_checkpoints_and_computes(
        self, mock_persist, mock_get_client
    ):
        pipeline = MagicMock(name="pipeline")

        dummy_stage = MagicMock()
        dummy_stage.name = "stage1"
        pipeline.executable_stages = [dummy_stage]

        # Create output object and checkpoint keys
        output = MagicMock()
        output.checkpoint_keys = ["vis1"]
        output.compute_tasks = ["task1"]
        output.__getitem__.return_value = "vis1_value"
        output.__setitem__ = MagicMock()
        output.compute_outputs = []
        output.stage_compute_tasks = []
        output.add_checkpoint_key = MagicMock()

        dummy_stage.return_value = output

        # Persist returns checkpoint and compute task values
        mock_persist.return_value = ["vis1_value", "task1"]

        scheduler = InstrumentalDaskRunner(_pipeline_=pipeline)
        scheduler.execute()

        # Check persist called for both stages
        mock_persist.assert_any_call(
            "vis1_value", "task1", optimize_graph=True
        )

        assert output.checkpoint_keys == []
        assert output.stage_compute_tasks == []

    @patch("ska_sdp_instrumental_calibration.scheduler.get_client")
    @patch("ska_sdp_instrumental_calibration.scheduler.futures_of")
    @patch("ska_sdp_instrumental_calibration.scheduler.as_completed")
    @patch("ska_sdp_instrumental_calibration.scheduler.dask.persist")
    def test_should_wait_and_throw_error_on_failure(
        self, mock_persist, mock_as_completed, mock_futures_of, get_client_mock
    ):
        pipeline = MagicMock(name="pipeline")

        dummy_stage = MagicMock()
        dummy_stage.name = "stage1"
        pipeline.executable_stages = [dummy_stage]

        # Create output object and checkpoint keys
        output = MagicMock()
        output.checkpoint_keys = ["vis1"]
        output.compute_tasks = ["task1"]
        output.__getitem__.return_value = "vis1_value"
        output.__setitem__ = MagicMock()
        output.compute_outputs = []
        output.stage_compute_tasks = []
        output.add_checkpoint_key = MagicMock()

        dummy_stage.return_value = output

        # Persist returns checkpoint and compute task values
        mock_persist.return_value = ["vis1_value", "task1"]

        error = Exception("Task failed")
        error_task = MagicMock()
        error_task.status = "error"
        error_task.result.side_effect = lambda: error
        success_task = MagicMock()
        success_task.status = "success"
        success_task.result.return_value = "ok"

        mock_delay1 = MagicMock()
        mock_delay2 = MagicMock()

        mock_futures_of.return_value = (mock_delay1, mock_delay2)
        mock_as_completed.return_value = [error_task, success_task]

        scheduler = InstrumentalDaskRunner(_pipeline_=pipeline)
        # Create mock tasks

        # Patch wait to return (done, not_done)
        with pytest.raises(Exception) as _error:
            scheduler.execute()
        assert _error.value is error_task.result()

        mock_futures_of.assert_called_once_with(["vis1_value", "task1"])
        mock_as_completed.assert_called_once_with((mock_delay1, mock_delay2))

    @patch("ska_sdp_instrumental_calibration.scheduler.get_client")
    @patch("ska_sdp_instrumental_calibration.scheduler.futures_of")
    @patch("ska_sdp_instrumental_calibration.scheduler.as_completed")
    @patch("ska_sdp_instrumental_calibration.scheduler.dask.persist")
    def test_should_wait_and_not_throw_error_on_success(
        self, mock_persist, mock_as_completed, mock_futures_of, get_client_mock
    ):
        pipeline = MagicMock(name="pipeline")

        dummy_stage = MagicMock()
        dummy_stage.name = "stage1"
        pipeline.executable_stages = [dummy_stage]

        # Create output object and checkpoint keys
        output = MagicMock()
        output.checkpoint_keys = ["vis1"]
        output.compute_tasks = ["task1"]
        output.__getitem__.return_value = "vis1_value"
        output.__setitem__ = MagicMock()
        output.compute_outputs = []
        output.stage_compute_tasks = []
        output.add_checkpoint_key = MagicMock()

        dummy_stage.return_value = output

        scheduler = InstrumentalDaskRunner(_pipeline_=pipeline)

        success_task = MagicMock()
        success_task.status = "success"
        success_task.result.return_value = "ok"

        mock_as_completed.return_value = [success_task]

        delayed_task = MagicMock()
        mock_futures_of.return_value = (delayed_task,)

        scheduler.execute()
        mock_as_completed.assert_called_once_with((delayed_task,))
