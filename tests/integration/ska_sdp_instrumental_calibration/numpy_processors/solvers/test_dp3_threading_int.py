import os

import pytest
from distributed import Client, LocalCluster

from ska_sdp_instrumental_calibration.numpy_processors.solvers.dp3_threading import (  # noqa: E501
    default_dp3_n_threads,
    get_dp3_lock,
)


@pytest.fixture(scope="module")
def dask_client():
    with LocalCluster(
        n_workers=1,
        threads_per_worker=3,
        processes=False,
        dashboard_address=None,
    ) as cluster, Client(cluster) as client:
        yield client


def test_default_dp3_n_threads_should_use_all_cpus_outside_worker():
    assert default_dp3_n_threads() == len(os.sched_getaffinity(0))


def test_default_dp3_n_threads_should_use_worker_nthreads(dask_client):
    assert dask_client.submit(default_dp3_n_threads).result() == 3


@pytest.mark.requires_dp3
def test_get_dp3_lock_should_be_shared_by_tasks_on_worker(dask_client):
    futures = [
        dask_client.submit(lambda _: id(get_dp3_lock()), idx, pure=False)
        for idx in range(6)
    ]

    lock_ids = dask_client.gather(futures)

    assert set(lock_ids) == {id(get_dp3_lock())}
