import pytest
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.scheduler import (
    DefaultScheduler,
    UpstreamOutput,
)


def test_should_return_manage_stage_count():
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


@patch("ska_sdp_instrumental_calibration.scheduler.futures_of")
@patch("ska_sdp_instrumental_calibration.scheduler.as_completed")
def test_should_wait_and_throw_error_on_failure(
    mock_as_completed, mock_futures_of
):
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

    scheduler = DefaultScheduler()
    # Create mock tasks

    # Patch wait to return (done, not_done)
    with pytest.raises(Exception) as error:
        scheduler.wait_and_throw_on_failure((mock_delay1, mock_delay2))
        assert error.value is error_task.result()

    mock_futures_of.assert_called_once_with((mock_delay1, mock_delay2))
    mock_as_completed.assert_called_once_with((mock_delay1, mock_delay2))


@patch("ska_sdp_instrumental_calibration.scheduler.futures_of")
@patch("ska_sdp_instrumental_calibration.scheduler.as_completed")
def test_should_wait_and_not_throw_error_on_success(
    mock_as_completed, mock_futures_of
):
    scheduler = DefaultScheduler()

    success_task = MagicMock()
    success_task.status = "success"
    success_task.result.return_value = "ok"

    mock_as_completed.return_value = [success_task]

    delayed_task = MagicMock()
    mock_futures_of.return_value = (delayed_task,)

    # Should not raise
    scheduler.wait_and_throw_on_failure((delayed_task,))
    mock_as_completed.assert_called_once_with((delayed_task,))


@patch("ska_sdp_instrumental_calibration.scheduler.get_client")
@patch("ska_sdp_instrumental_calibration.scheduler.dask.persist")
@patch.object(DefaultScheduler, "wait_and_throw_on_failure")
def test_should_wait_to_persist_when_client_present(
    mock_wait_and_throw, mock_persist, mock_get_client
):
    # Simulate client is present
    mock_get_client.return_value = MagicMock()
    dummy_stage = MagicMock()
    dummy_stage.name = "stage1"

    mock_vis = MagicMock()

    def side_effect(output):
        output["vis"] = mock_vis
        output.add_checkpoint_key("vis")
        return output

    dummy_stage.side_effect = side_effect

    # Simulate persisted values (one failed task, one success)
    failed_task = MagicMock()
    success_task = MagicMock()
    mock_persist.return_value = [failed_task, success_task]

    # Simulate wait_and_throw_on_failure raises an exception
    error = RuntimeError("Task failed")
    mock_wait_and_throw.side_effect = error

    scheduler = DefaultScheduler()
    with pytest.raises(Exception) as excinfo:
        scheduler.schedule([dummy_stage])
    assert excinfo.value is error

    # Ensure persist called with correct arguments
    mock_persist.assert_called_once_with(mock_vis, optimize_graph=True)
    # Ensure wait_and_throw_on_failure called with persisted values
    mock_wait_and_throw.assert_called_once_with([failed_task, success_task])


@patch("ska_sdp_instrumental_calibration.scheduler.get_client")
@patch("ska_sdp_instrumental_calibration.scheduler.dask.persist")
@patch.object(DefaultScheduler, "wait_and_throw_on_failure")
def test_should_not_wait_to_persist_with_no_client(
    mock_wait_and_throw, mock_persist, mock_get_client
):
    # Simulate client is not present
    mock_get_client.side_effect = Exception("No client")
    dummy_stage = MagicMock()
    dummy_stage.name = "stage1"

    mock_vis = MagicMock()

    def side_effect(output):
        output["vis"] = mock_vis
        output.add_checkpoint_key("vis")
        return output

    dummy_stage.side_effect = side_effect

    # Simulate persisted values (one failed task, one success)
    failed_task = MagicMock()
    success_task = MagicMock()
    mock_persist.return_value = [failed_task, success_task]

    scheduler = DefaultScheduler()
    scheduler.schedule([dummy_stage])

    # Ensure persist called with correct arguments
    mock_persist.assert_called_once_with(mock_vis, optimize_graph=True)
    # ensure wait is not called
    mock_wait_and_throw.assert_not_called()
