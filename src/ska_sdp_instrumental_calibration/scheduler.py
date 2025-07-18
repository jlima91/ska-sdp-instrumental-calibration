from functools import reduce

from ska_sdp_piper.piper.scheduler import PiperScheduler


class UpstreamOutput:
    def __init__(self):
        self.__stage_outputs = {}
        self.__compute_tasks = []
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
        return self.__compute_tasks

    def add_compute_tasks(self, *args):
        self.__compute_tasks.extend(args)


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
        self._stage_outputs = reduce(
            lambda output, stage: stage(output), stages, self._stage_outputs
        )

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
