import dask
import pytest
from distributed import Client
from mock import MagicMock
from ska_sdp_piper.piper.stage import Stage

from ska_sdp_instrumental_calibration.scheduler import InstrumentalDaskRunner


@pytest.fixture(scope="function")
def passing_pipeline():
    pipeline = MagicMock(name="pipeline")
    successful_task = dask.delayed(lambda: 42)()

    # create dummy stage with two tasks
    def stage_definition(_upstream_output_):
        _upstream_output_.add_compute_tasks(successful_task)
        return _upstream_output_

    stage = Stage(
        name="test_stage",
        stage_definition=stage_definition,
    )
    pipeline.executable_stages = [stage]

    yield pipeline


@pytest.fixture(scope="function")
def failing_pipeline():
    pipeline = MagicMock(name="pipeline")
    successful_task = dask.delayed(lambda: 42)()
    failing_task = dask.delayed(
        lambda: 1 / 0
    )()  # This will raise ZeroDivisionError

    # create dummy stage with two tasks
    def stage_definition(_upstream_output_):
        _upstream_output_.add_compute_tasks(successful_task, failing_task)
        return _upstream_output_

    stage = Stage(
        name="test_stage",
        stage_definition=stage_definition,
    )
    pipeline.executable_stages = [stage]

    yield pipeline


def test_should_scheduler_wait_and_fail_for_failed_tasks_with_client(
    failing_pipeline,
):
    # Create a mix of successful and failing tasks
    with Client() as _:
        default_scheduler = InstrumentalDaskRunner(failing_pipeline)

        with pytest.raises(ZeroDivisionError):
            default_scheduler.execute()


def test_should_scheduler_wait_and_success_with_client(passing_pipeline):
    # Create a mix of successful and failing tasks
    with Client() as _:
        default_scheduler = InstrumentalDaskRunner(passing_pipeline)
        default_scheduler.execute()


def test_should_scheduler_wait_and_fail_for_failed_tasks_without_client(
    failing_pipeline,
):
    # Create a mix of successful and failing tasks
    default_scheduler = InstrumentalDaskRunner(failing_pipeline)

    with pytest.raises(ZeroDivisionError):
        default_scheduler.execute()


def test_should_scheduler_wait_and_success_without_client(passing_pipeline):
    default_scheduler = InstrumentalDaskRunner(passing_pipeline)

    default_scheduler.execute()
