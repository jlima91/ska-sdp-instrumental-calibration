import pytest
from mock import MagicMock
from ska_sdp_piper.piper.stage import Stage

from ska_sdp_instrumental_calibration.scheduler.scheduler import (
    InstrumentalDaskRunner,
    delayed,
)


@pytest.fixture(scope="function")
def passing_pipeline():
    pipeline = MagicMock(name="pipeline")
    successful_task = delayed(lambda: 42)()

    # create dummy stage with two tasks
    def stage_definition(_upstream_output_):
        successful_task()
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
    successful_task = delayed(lambda: 42)
    failing_task = delayed(lambda: 1 / 0)  # This will raise ZeroDivisionError

    # create dummy stage with two tasks
    def stage_definition(_upstream_output_):
        successful_task()
        failing_task()
        return _upstream_output_

    stage = Stage(
        name="test_stage",
        stage_definition=stage_definition,
    )
    pipeline.executable_stages = [stage]

    yield pipeline


def test_should_scheduler_wait_and_fail_for_failed_tasks_without_client(
    failing_pipeline,
):
    # Create a mix of successful and failing tasks
    default_scheduler = InstrumentalDaskRunner(_pipeline_=failing_pipeline)

    with pytest.raises(ZeroDivisionError):
        default_scheduler.execute()


def test_should_scheduler_wait_and_success_without_client(passing_pipeline):
    default_scheduler = InstrumentalDaskRunner(_pipeline_=passing_pipeline)
    default_scheduler.execute()
