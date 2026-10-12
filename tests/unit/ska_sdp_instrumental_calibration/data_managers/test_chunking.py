import logging

import pytest
from mock import Mock, patch

from ska_sdp_instrumental_calibration.data_managers.chunking import (
    ClusterSpec,
    estimate_channels_per_chunk,
    get_cluster_spec,
)

_GIB = 2**30

_GET_CLIENT_PATCH = (
    "ska_sdp_instrumental_calibration.data_managers.chunking"
    ".distributed.get_client"
)


@patch(
    "ska_sdp_instrumental_calibration.data_managers.chunking.MEMORY_LIMIT",
    64 * _GIB,
)
@patch("ska_sdp_instrumental_calibration.data_managers.chunking.CPU_COUNT", 8)
@patch(_GET_CLIENT_PATCH, side_effect=ValueError("No global client found"))
def test_should_use_local_machine_resources_when_no_client(get_client_mock):
    actual_output = get_cluster_spec()

    assert actual_output == ClusterSpec(
        total_threads=8, memory_per_thread_bytes=8 * _GIB
    )


@patch(_GET_CLIENT_PATCH)
def test_should_use_least_memory_per_thread_across_client_workers(
    get_client_mock,
):
    client_mock = Mock(name="client")
    client_mock.scheduler_info.return_value = {
        "workers": {
            "tcp://worker-a": {"nthreads": 2, "memory_limit": 8 * _GIB},
            "tcp://worker-b": {"nthreads": 2, "memory_limit": 8 * _GIB},
        }
    }
    get_client_mock.return_value = client_mock

    actual_output = get_cluster_spec()

    client_mock.scheduler_info.assert_called_once_with(n_workers=-1)
    assert actual_output == ClusterSpec(
        total_threads=4, memory_per_thread_bytes=4 * _GIB
    )


@patch(
    "ska_sdp_instrumental_calibration.data_managers.chunking.MEMORY_LIMIT",
    16 * _GIB,
)
@patch(_GET_CLIENT_PATCH)
def test_should_use_system_memory_for_workers_without_memory_limit(
    get_client_mock,
):
    client_mock = Mock(name="client")
    client_mock.scheduler_info.return_value = {
        "workers": {"tcp://worker-a": {"nthreads": 4, "memory_limit": 0}}
    }
    get_client_mock.return_value = client_mock

    actual_output = get_cluster_spec()

    assert actual_output == ClusterSpec(
        total_threads=4, memory_per_thread_bytes=4 * _GIB
    )


@patch(_GET_CLIENT_PATCH)
def test_should_raise_runtime_error_when_cluster_has_no_workers(
    get_client_mock,
):
    client_mock = Mock(name="client")
    client_mock.scheduler_info.return_value = {"workers": {}}
    get_client_mock.return_value = client_mock

    with pytest.raises(RuntimeError, match="Dask cluster has no workers."):
        get_cluster_spec()


class TestEstimateChannelsPerChunk:
    def test_should_limit_tasks_to_three_per_thread(self):
        cluster_spec = ClusterSpec(
            total_threads=96, memory_per_thread_bytes=8 * _GIB
        )

        # 208 times x 6 baselines x 4 pols, 55380 channels: ~0.1 MiB/channel
        actual_output = estimate_channels_per_chunk(
            55380, 208 * 6 * 4, cluster_spec
        )

        # ceil(55380 / (96 * 3))
        assert actual_output == 193

    def test_should_not_create_chunks_smaller_than_minimum_size(self):
        cluster_spec = ClusterSpec(
            total_threads=96, memory_per_thread_bytes=8 * _GIB
        )

        actual_output = estimate_channels_per_chunk(10000, 1000, cluster_spec)

        # ceil(16 MiB / (1000 * 21 B))
        assert actual_output == 799

    def test_should_not_create_chunks_larger_than_maximum_size(self):
        cluster_spec = ClusterSpec(
            total_threads=1, memory_per_thread_bytes=100 * _GIB
        )

        actual_output = estimate_channels_per_chunk(
            1000, 5_000_000, cluster_spec
        )

        # floor(1 GiB / (5e6 * 21 B))
        assert actual_output == 10

    def test_should_not_create_chunks_larger_than_memory_per_thread(self):
        cluster_spec = ClusterSpec(
            total_threads=1, memory_per_thread_bytes=1 * _GIB
        )

        actual_output = estimate_channels_per_chunk(
            1000, 5_000_000, cluster_spec
        )

        # floor(0.6 * 1 GiB / (5e6 * 21 B))
        assert actual_output == 6

    def test_should_not_exceed_number_of_channels(self):
        cluster_spec = ClusterSpec(
            total_threads=96, memory_per_thread_bytes=8 * _GIB
        )

        actual_output = estimate_channels_per_chunk(10, 1000, cluster_spec)

        assert actual_output == 10

    def test_should_return_one_channel_with_warning_when_channel_exceeds_max(
        self, caplog
    ):
        cluster_spec = ClusterSpec(
            total_threads=1, memory_per_thread_bytes=10 * _GIB
        )

        with caplog.at_level(logging.WARNING):
            actual_output = estimate_channels_per_chunk(
                100, 60_000_000, cluster_spec
            )

        assert actual_output == 1
        assert "Using 1 channel per chunk." in caplog.text

    def test_should_raise_value_error_when_channel_exceeds_thread_memory(
        self,
    ):
        cluster_spec = ClusterSpec(
            total_threads=1, memory_per_thread_bytes=100 * 2**20
        )

        with pytest.raises(
            ValueError,
            match=r"A single frequency channel needs 100\.14 MiB, but a "
            r"thread can only hold 60\.00 MiB\.",
        ):
            estimate_channels_per_chunk(1000, 5_000_000, cluster_spec)
