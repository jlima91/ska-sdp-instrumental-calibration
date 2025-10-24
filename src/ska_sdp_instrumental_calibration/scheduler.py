import logging

import dask
from distributed import get_client, wait
from ska_sdp_piper.piper.scheduler import PiperScheduler

logger = logging.getLogger()


class UpstreamOutput:
    def __init__(self):
        self.__stage_outputs = {}
        self.stage_compute_tasks = []
        self.checkpoint_keys = []
        self.compute_outputs = []
        self.__call_count = {}

    def __setitem__(self, key, value):
        self.__stage_outputs[key] = value

    def __getitem__(self, key):
        if key not in self.__stage_outputs:
            raise AttributeError(f"{key} not present in upstream-output.")
        return self.__stage_outputs[key]

    def __getattr__(self, key):
        if key not in self.__stage_outputs:
            raise AttributeError(f"{key} not present in upstream-output.")
        return self.__stage_outputs[key]

    def __contains__(self, key):
        return key in self.__stage_outputs

    def get_call_count(self, stage_name):
        return self.__call_count.get(stage_name, 0)

    def increment_call_count(self, stage_name):
        self.__call_count[stage_name] = (
            self.__call_count.get(stage_name, 0) + 1
        )

    @property
    def compute_tasks(self):
        return self.stage_compute_tasks

    def add_compute_tasks(self, *args):
        self.stage_compute_tasks.extend(args)

    def add_checkpoint_key(self, *args):
        self.checkpoint_keys.extend(args)


class DefaultScheduler(PiperScheduler):
    """
    Schedules and executes dask wrapped functions on the local machine

    Attributes
    ----------
    _tasks: list(Delayed)
        Dask delayed outputs from the scheduled tasks
    """

    def __init__(self):
        self._stage_outputs = UpstreamOutput()

    def schedule(self, stages):
        """
        Schedules the stages as dask delayed objects

        Parameters
        ----------
          stages: list(stages.Stage)
            List of stages to schedule
        """
        is_client_present = False

        try:
            get_client()
            is_client_present = True
        except Exception:
            pass

        output = self._stage_outputs
        for stage in stages:
            logger.info(
                f"Starting {stage.name}",
                extra={"tags": f"sdpPhase:{stage.name.upper()},state:START"},
            )

            output = stage(output)

            checkpoints = [output[key] for key in output.checkpoint_keys]
            persisted_values = dask.persist(
                *(checkpoints + output.compute_tasks), optimize_graph=True
            )

            for idx, key in enumerate(output.checkpoint_keys):
                output[key] = persisted_values[idx]

            output.compute_outputs += persisted_values[
                len(output.checkpoint_keys) :  # noqa:E203
            ]

            if is_client_present:
                wait(persisted_values)

            output.checkpoint_keys = []
            output.stage_compute_tasks = []

            logger.info(
                f"Finished {stage.name}",
                extra={
                    "tags": f"sdpPhase:{stage.name.upper()},state:FINISHED"
                },
            )

        self._stage_outputs = output

    def append(self, task):
        """
        Appends a dask task to the task list

        Parameters
        ----------
          task: Delayed
            Dask delayed object
        """
        self._stage_outputs.add_compute_tasks(task)

    def extend(self, tasks):
        """
        Extends the task list with a list of dask tasks

        Parameters
        ----------
          task: list(Delayed)
            Dask delayed objects
        """

        self._stage_outputs.add_compute_tasks(*tasks)

    @property
    def tasks(self):
        """
        Returns all the delayed task

        Returns
        -------
            list(Delayed)
        """
        return self._stage_outputs.compute_tasks
