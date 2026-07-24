from mock import MagicMock, call, patch

from ska_sdp_instrumental_calibration.scheduler.deferred_tasks import (
    DeferredTask,
)


@patch(
    "ska_sdp_instrumental_calibration.scheduler.deferred_tasks"
    ".unpack_collections"
)
def test_should_create_lazy_params(unpack_collection_mock):

    func = MagicMock(name="func")
    unpack_collection_mock.side_effect = [
        ("lazy_args", "args_repack"),
        ("lazy_kwargs", "kwargs_repack"),
    ]
    def_task = DeferredTask(func, "arg1", "arg2", kwarg="kwarg")

    unpack_collection_mock.assert_has_calls(
        [call(("arg1", "arg2")), call({"kwarg": "kwarg"})]
    )

    assert def_task.params == {
        "args": "lazy_args",
        "kwargs": "lazy_kwargs",
    }

    def_task()

    func.assert_called_once_with("arg1", "arg2", kwarg="kwarg")


@patch(
    "ska_sdp_instrumental_calibration.scheduler.deferred_tasks"
    ".unpack_collections"
)
@patch(
    "ska_sdp_instrumental_calibration.scheduler.deferred_tasks.dask.delayed"
)
def test_should_return_delayed_object(delayed_mock, unpack_collection_mock):

    delayed_func = MagicMock(name="delayed_func")
    repack_mock1 = MagicMock(
        name="repack_mock1", return_value=(("p_arg1", "p_arg2"),)
    )
    repack_mock2 = MagicMock(
        name="repack_mock2", return_value=({"key": "p_kwarg"},)
    )

    delayed_mock.return_value = delayed_func

    unpack_collection_mock.side_effect = [
        ("lazy_args", repack_mock1),
        ("lazy_kwargs", repack_mock2),
    ]
    def_task = DeferredTask("func", "arg1", "arg2", kwarg="kwarg")

    result = def_task.delayed(args=("arg1", "arg2"), kwargs={"key": "kwarg"})
    delayed_mock.assert_called_once_with("func")

    repack_mock1.assert_called_once_with(("arg1", "arg2"))
    repack_mock2.assert_called_once_with({"key": "kwarg"})

    delayed_func.assert_called_once_with("p_arg1", "p_arg2", key="p_kwarg")
    assert result == delayed_func.return_value


@patch(
    "ska_sdp_instrumental_calibration.scheduler.deferred_tasks"
    ".unpack_collections",
    return_value=["arg", "r_arg"],
)
@patch("ska_sdp_instrumental_calibration.scheduler.deferred_tasks.tokenize")
def test_should_be_hashable(tokenize_mock, unpack_collection_mock):
    def_task = DeferredTask("func", "arg1", "arg2", kwarg="kwarg")
    def_task_2 = DeferredTask("func", "arg1", "arg2", kwarg="kwarg")
    assert hash(def_task) == hash(tokenize_mock.return_value)

    assert def_task == def_task_2
    assert not def_task == "DeferredTask"
