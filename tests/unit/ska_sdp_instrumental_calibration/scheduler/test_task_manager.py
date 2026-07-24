from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.scheduler.task_manager import (
    _TaskManager,
)


def test_should_register_task():
    task = MagicMock(name="task")
    task2 = MagicMock(name="task2")

    task_manager = _TaskManager()

    task_manager.register(task)
    task_manager.register(task2)

    assert task_manager._tracked_arrays == {
        task: task.params,
        task2: task2.params,
    }


@patch(
    "ska_sdp_instrumental_calibration.scheduler.task_manager.dask.persist",
)
@patch("ska_sdp_instrumental_calibration.scheduler.task_manager.dask.compute")
def test_should_compute_task(compute_mock, persist_mock):
    task = MagicMock(name="task")
    task2 = MagicMock(name="task2")

    persist_mock.return_value = [
        {
            task: {"key": "p_arg_task"},
            task2: {"key": "p_arg_task2"},
        }
    ]

    task_manager = _TaskManager()

    task_manager.register(task)
    task_manager.register(task2)

    res = task_manager.compute()

    persist_mock.assert_called_once_with(
        {
            task: task.params,
            task2: task2.params,
        },
        optimize_graph=True,
    )

    task.delayed.assert_called_once_with(key="p_arg_task")
    task2.delayed.assert_called_once_with(key="p_arg_task2")

    compute_mock.assert_called_once_with(
        task.delayed.return_value, task2.delayed.return_value
    )

    assert res == compute_mock.return_value

    assert not task_manager._tracked_arrays
