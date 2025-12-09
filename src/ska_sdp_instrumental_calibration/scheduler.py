import logging

import dask
from distributed import get_client, wait
from ska_sdp_piper.piper.scheduler import PiperScheduler

logger = logging.getLogger()


class UpstreamOutput:
    """
    Container for managing outputs and metadata between pipeline stages.

    This class acts as a shared context object, allowing downstream stages to
    access the results of upstream stages via dictionary-style or attribute-
    style access. It also tracks computational tasks, checkpoint keys, and
    execution counts for each stage.

    Attributes
    ----------
    stage_compute_tasks : list
        A list of delayed compute tasks (e.g., Dask graphs) accumulated
        during the pipeline execution.
    checkpoint_keys : list
        A list of keys identifying data that should be checkpointed or
        persisted.
    compute_outputs : list
        A list to store results of computations.
    """

    def __init__(self):
        """
        Initialize the UpstreamOutput container.
        """
        self.__stage_outputs = {}
        self.stage_compute_tasks = []
        self.checkpoint_keys = []
        self.compute_outputs = []
        self.__call_count = {}

    def __setitem__(self, key, value):
        """
        Store an output value for a specific stage key.

        Parameters
        ----------
        key : str
            The identifier for the stage output.
        value : any
            The data or object to store.
        """
        self.__stage_outputs[key] = value

    def __getitem__(self, key):
        """
        Retrieve a stage output by key.

        Parameters
        ----------
        key : str
            The identifier of the output to retrieve.

        Returns
        -------
        any
            The value associated with the key.

        Raises
        ------
        AttributeError
            If the key is not present in the outputs.
        """
        if key not in self.__stage_outputs:
            raise AttributeError(f"{key} not present in upstream-output.")
        return self.__stage_outputs[key]

    def __getattr__(self, key):
        """
        Retrieve a stage output using attribute access syntax.

        Parameters
        ----------
        key : str
            The identifier of the output to retrieve.

        Returns
        -------
        any
            The value associated with the key.

        Raises
        ------
        AttributeError
            If the key is not present in the outputs.
        """
        if key not in self.__stage_outputs:
            raise AttributeError(f"{key} not present in upstream-output.")
        return self.__stage_outputs[key]

    def __contains__(self, key):
        """
        Check if a key exists in the stage outputs.

        Parameters
        ----------
        key : str
            The identifier to check.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return key in self.__stage_outputs

    def get_call_count(self, stage_name):
        """
        Get the number of times a specific stage has been executed.

        Parameters
        ----------
        stage_name : str
            The name of the stage.

        Returns
        -------
        int
            The execution count (default is 0).
        """
        return self.__call_count.get(stage_name, 0)

    def increment_call_count(self, stage_name):
        """
        Increment the execution counter for a specific stage.

        Parameters
        ----------
        stage_name : str
            The name of the stage to increment.
        """
        self.__call_count[stage_name] = (
            self.__call_count.get(stage_name, 0) + 1
        )

    @property
    def compute_tasks(self):
        """
        list: Get the list of accumulated compute tasks.
        """
        return self.stage_compute_tasks

    def add_compute_tasks(self, *args):
        """
        Register new compute tasks to the pipeline.

        Parameters
        ----------
        *args
            One or more task objects (e.g., Dask delayed objects) to add
            to the execution queue.
        """
        self.stage_compute_tasks.extend(args)

    def add_checkpoint_key(self, *args):
        """
        Register keys that should be checkpointed.

        Parameters
        ----------
        *args
            One or more string keys identifying outputs that require
            checkpointing or persistence.
        """
        self.checkpoint_keys.extend(args)


class DefaultScheduler(PiperScheduler):
    """
    Schedules and executes Dask-wrapped functions for pipeline stages.

    This scheduler manages the execution of a list of `Stage` objects. It
    handles the chaining of outputs between stages (via `UpstreamOutput`),
    manages Dask persistence for checkpoints, and coordinates execution
    via a local or distributed Dask client.

    Attributes
    ----------
    _stage_outputs : UpstreamOutput
        Container for stage results, checkpoints, and pending tasks.
    """

    def __init__(self):
        """
        Initialize the default scheduler.
        """
        self._stage_outputs = UpstreamOutput()

    def schedule(self, stages):
        """
        Execute the provided list of pipeline stages.

        Iterates through the stages, executing each one sequentially. For
        each stage, it:

        1. Logs the start of the stage.
        2. Invokes the stage callable with the current upstream outputs.
        3. Persists any data flagged for checkpointing and any accumulated
           compute tasks using `dask.persist`.
        4. Updates the output container with the persisted results.
        5. Waits for completion if a Dask client is present.
        6. Logs the completion of the stage.

        Parameters
        ----------
        stages : list of Stage
            A list of stage objects to be executed.
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
        Append a single Dask task to the current execution queue.

        Parameters
        ----------
        task : dask.delayed.Delayed
            The Dask delayed object representing the task.
        """
        self._stage_outputs.add_compute_tasks(task)

    def extend(self, tasks):
        """
        Extend the execution queue with a list of Dask tasks.

        Parameters
        ----------
        tasks : list of dask.delayed.Delayed
            A list of Dask delayed objects to add.
        """

        self._stage_outputs.add_compute_tasks(*tasks)

    @property
    def tasks(self):
        """
        list of dask.delayed.Delayed: Get the list of pending compute tasks.
        """
        return self._stage_outputs.compute_tasks
