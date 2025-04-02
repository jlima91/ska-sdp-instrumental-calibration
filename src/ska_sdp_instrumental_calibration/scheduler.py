from functools import reduce

from ska_sdp_piper.piper.scheduler import PiperScheduler


class DefaultScheduler(PiperScheduler):
    """
    Schedules and executes dask wrapped functions on the local machine

    Attributes
    ----------
    _tasks: list(Delayed)
        Dask delayed outputs from the scheduled tasks
    """

    def __init__(self):
        self._stage_outputs = {}

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

    @property
    def tasks(self):
        """
        Returns all the delayed task

        Returns
        -------
            list(Delayed)
        """
        return []
