import logging
import math
from dataclasses import dataclass

import distributed
from dask.system import CPU_COUNT
from distributed.system import MEMORY_LIMIT

logger = logging.getLogger(__name__)

MIB = 1024 * 1024

# Bytes held per visibility sample by one task:
# vis (complex64) + modelvis (complex64) + weight (float32) + flag (bool)
BYTES_PER_VISIBILITY = 8 + 8 + 4 + 1

MIN_CHUNK_SIZE_MIB = 16
MAX_CHUNK_SIZE_MIB = 1024
TASKS_FACTOR = 3
# Fraction of thread memory usable by chunk data, leaving room for
# dask overheads and intermediate arrays
BREATHING_SPACE = 0.6


@dataclass(frozen=True)
class ClusterSpec:
    """
    Compute resources available to the pipeline.

    Parameters
    ----------
    total_threads: int
        Total number of threads across all workers.
    memory_per_thread_bytes: int
        Memory available to a single thread in bytes.
    """

    total_threads: int
    memory_per_thread_bytes: int


def get_cluster_spec() -> ClusterSpec:
    """
    Get the compute resources from the active dask distributed client.
    If no client is active, the resources of the local machine are used,
    as dask runs on a local thread pool.

    For a heterogeneous cluster, memory per thread is taken from the worker
    with the least memory per thread.

    Returns
    -------
    ClusterSpec
        Compute resources available to the pipeline.
    """
    try:
        client = distributed.get_client()
    except ValueError:
        logger.info("No dask client found. Using local machine resources.")
        return ClusterSpec(
            total_threads=CPU_COUNT,
            memory_per_thread_bytes=MEMORY_LIMIT // CPU_COUNT,
        )

    workers = client.scheduler_info(n_workers=-1)["workers"].values()
    if not workers:
        raise RuntimeError("Dask cluster has no workers.")

    return ClusterSpec(
        total_threads=sum(worker["nthreads"] for worker in workers),
        memory_per_thread_bytes=min(
            (worker["memory_limit"] or MEMORY_LIMIT) // worker["nthreads"]
            for worker in workers
        ),
    )


def estimate_channels_per_chunk(
    nchannels: int,
    nvis_per_channel: int,
    cluster_spec: ClusterSpec,
) -> int:
    """
    Estimate the number of frequency channels per chunk.

    The estimate aims for at most ``TASKS_FACTOR`` tasks per thread, while
    keeping each chunk between ``MIN_CHUNK_SIZE_MIB`` and
    ``MAX_CHUNK_SIZE_MIB``, and within the memory available to one thread.
    Chunk sizes include visibilities, model visibilities, weights and flags.

    Parameters
    ----------
    nchannels: int
        Number of frequency channels in the visibility.
    nvis_per_channel: int
        Number of visibility samples in one frequency channel,
        i.e. ntimes * nbaselines * npolarisations.
    cluster_spec: ClusterSpec
        Compute resources available to the pipeline.

    Returns
    -------
    int
        Number of frequency channels per chunk.

    Raises
    ------
    ValueError
        If a single frequency channel does not fit in the memory available
        to one thread.
    """
    bytes_per_channel = nvis_per_channel * BYTES_PER_VISIBILITY
    mem_per_thread = math.ceil(
        cluster_spec.memory_per_thread_bytes * BREATHING_SPACE
    )

    if bytes_per_channel > mem_per_thread:
        raise ValueError(
            f"A single frequency channel needs {bytes_per_channel / MIB:.2f}"
            f" MiB, but a thread can only hold {mem_per_thread / MIB:.2f} "
            "MiB. Increase the memory per thread of the dask cluster."
        )

    if bytes_per_channel >= MAX_CHUNK_SIZE_MIB * MIB:
        logger.warning(
            "A single frequency channel needs %.2f MiB, which exceeds the "
            "maximum chunk size of %d MiB. Using 1 channel per chunk.",
            bytes_per_channel / MIB,
            MAX_CHUNK_SIZE_MIB,
        )
        return 1

    max_tasks = cluster_spec.total_threads * TASKS_FACTOR

    nchan = max(1, math.ceil(nchannels / max_tasks))
    # Avoid tiny chunks, whose scheduling overhead outweighs their work
    nchan = max(nchan, math.ceil(MIN_CHUNK_SIZE_MIB * MIB / bytes_per_channel))
    nchan = min(nchan, (MAX_CHUNK_SIZE_MIB * MIB) // bytes_per_channel)
    nchan = min(nchan, mem_per_thread // bytes_per_channel)

    return min(nchan, nchannels)
