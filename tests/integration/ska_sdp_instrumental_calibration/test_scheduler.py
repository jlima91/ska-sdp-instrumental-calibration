import dask
import pytest
from distributed import Client
from ska_sdp_piper.piper.configurations import Configuration
from ska_sdp_piper.piper.stage import Stage

from ska_sdp_instrumental_calibration.scheduler import DefaultScheduler


def test_should_scheduler_wait_and_fail_for_failed_tasks_with_client():
    # Create a mix of successful and failing tasks
    with Client() as _:
        successful_task = dask.delayed(lambda: 42)()
        failing_task = dask.delayed(
            lambda: 1 / 0
        )()  # This will raise ZeroDivisionError

        # create dummy stage with two tasks
        def stage_definition(output):
            output.add_compute_tasks(successful_task, failing_task)
            return output

        stage = Stage(
            name="test_stage",
            stage_definition=stage_definition,
            configuration=Configuration(),
        )
        stage.add_additional_parameters(param=1)

        default_scheduler = DefaultScheduler()

        with pytest.raises(ZeroDivisionError):
            default_scheduler.schedule([stage])


def test_should_scheduler_wait_and_success_with_client():
    # Create a mix of successful and failing tasks
    with Client() as _:
        successful_task = dask.delayed(lambda: 42)()

        # create dummy stage with two tasks
        def stage_definition(output):
            output.add_compute_tasks(successful_task)
            return output

        stage = Stage(
            name="test_stage",
            stage_definition=stage_definition,
            configuration=Configuration(),
        )
        stage.add_additional_parameters(param=1)

        default_scheduler = DefaultScheduler()

        default_scheduler.schedule([stage])


def test_should_scheduler_wait_and_fail_for_failed_tasks_without_client():
    # Create a mix of successful and failing tasks
    successful_task = dask.delayed(lambda: 42)()
    failing_task = dask.delayed(
        lambda: 1 / 0
    )()  # This will raise ZeroDivisionError

    # create dummy stage with two tasks
    def stage_definition(output):
        output.add_compute_tasks(successful_task, failing_task)
        return output

    stage = Stage(
        name="test_stage",
        stage_definition=stage_definition,
        configuration=Configuration(),
    )
    stage.add_additional_parameters(param=1)

    default_scheduler = DefaultScheduler()

    with pytest.raises(ZeroDivisionError):
        default_scheduler.schedule([stage])


def test_should_scheduler_wait_and_success_without_client():
    # Create a mix of successful and failing tasks
    successful_task = dask.delayed(lambda: 42)()

    # create dummy stage with two tasks
    def stage_definition(output):
        output.add_compute_tasks(successful_task)
        return output

    stage = Stage(
        name="test_stage",
        stage_definition=stage_definition,
        configuration=Configuration(),
    )
    stage.add_additional_parameters(param=1)

    default_scheduler = DefaultScheduler()

    default_scheduler.schedule([stage])
