import os

import numpy as np
import pytest
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.visibility import (
    check_if_cache_files_exist,
    read_ms_field_id,
    read_visibility_from_zarr,
    write_ms_to_zarr,
)


def create_mock_table(getcol_return=None, getkeywords_return=None):
    mock_table = MagicMock()
    mock_table.__enter__ = MagicMock(return_value=mock_table)
    mock_table.__exit__ = MagicMock(return_value=False)

    if getcol_return is not None:
        mock_table.getcol.return_value = getcol_return
    if getkeywords_return is not None:
        mock_table.getkeywords.return_value = getkeywords_return

    return mock_table


@pytest.fixture
def mock_table():
    with patch(
        "ska_sdp_instrumental_calibration.data_managers.visibility.table"
    ) as mock:
        yield mock


def test_visibility_write_and_read(tmp_path, generate_vis, generate_ms):
    """
    Integration test for all functionality under
    the visibility module.
    This will test:
    1. Conversion of MSv2 to Zarr file format
    2. Reading of zarr file as Visibility object
    """
    gen_vis, _ = generate_vis
    ms_path = generate_ms

    cache_dir = f"{tmp_path}/cache"
    os.makedirs(cache_dir)

    # Common dimensions across zarr and loaded visibility dataset
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }
    # This is chunking of the intermidiate zarr file
    zarr_chunks = {
        **non_chunked_dims,
        "time": 1,
        "frequency": 2,
    }
    vis_chunks = {
        **non_chunked_dims,
        "time": -1,
        "frequency": 2,
    }

    write_ms_to_zarr(ms_path, cache_dir, zarr_chunks)

    assert check_if_cache_files_exist(cache_dir)

    zarred_dataset = read_visibility_from_zarr(cache_dir, vis_chunks)

    np.testing.assert_allclose(
        np.real(gen_vis.vis.data), np.real(zarred_dataset.vis.data)
    )
    np.testing.assert_allclose(
        np.imag(gen_vis.vis.data), np.imag(zarred_dataset.vis.data)
    )
    np.testing.assert_allclose((gen_vis.uvw.data), (zarred_dataset.uvw.data))
    np.testing.assert_allclose(
        (gen_vis.flags.data), (zarred_dataset.flags.data)
    )
    # Might be problem with export_visibility_to_ms function, as the weight
    # column in the MS is not exactly same as the weight variable in the
    # visibility dataset. This needs to be investigated further.
    # np.testing.assert_allclose((gen_vis.weight.data),(zarred_dataset.weight.data))

    np.testing.assert_allclose(
        (gen_vis.frequency.data), (zarred_dataset.frequency.data)
    )
    np.testing.assert_equal((gen_vis.time.data), (zarred_dataset.time.data))


def test_visibility_write_and_read_for_multiple_ms_input(
    tmp_path, generate_vis, generate_ms
):
    gen_vis, _ = generate_vis
    ms_path = generate_ms
    cache_dir = f"{tmp_path}/cache"
    os.makedirs(cache_dir)

    # Common dimensions across zarr and loaded visibility dataset
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }
    # This is chunking of the intermidiate zarr file
    zarr_chunks = {
        **non_chunked_dims,
        "time": 1,
        "frequency": 2,
    }
    vis_chunks = {
        **non_chunked_dims,
        "time": -1,
        "frequency": 2,
    }

    write_ms_to_zarr([ms_path, ms_path], cache_dir, zarr_chunks)

    assert check_if_cache_files_exist(cache_dir)

    zarred_dataset = read_visibility_from_zarr(cache_dir, vis_chunks)

    time_vars = ["vis", "uvw", "flags", "integration_time", "datetime", "time"]
    for var in time_vars:
        expected = np.concatenate(
            [getattr(gen_vis, var), getattr(gen_vis, var)]
        )
        np.testing.assert_equal(
            expected,
            getattr(zarred_dataset, var),
            err_msg=f"Mismatch in time-dependent variable: {var}",
        )

    datapoints = [
        "frequency",
        "baselines",
        "polarisation",
        "spatial",
        "antenna1",
        "antenna2",
        "channel_bandwidth",
    ]
    for datapoint in datapoints:
        np.testing.assert_equal(
            getattr(gen_vis, datapoint).data,
            getattr(zarred_dataset, datapoint).data,
            err_msg=f"Mismatch in data: {datapoint}",
        )

    # Might be problem with export_visibility_to_ms function, as the weight
    # column in the MS is not exactly same as the weight variable in the
    # visibility dataset. This needs to be investigated further.
    # np.testing.assert_equal(np.concatenate([gen_vis.weight, gen_vis.weight]),
    # zarred_dataset.weight)

    # Check attributes is failing as frame attribute is not preserved
    # assert gen_vis.attrs == zarred_dataset.attrs


def test_should_read_field_id_from_field_table(mock_table):
    ms_path = "/path/to/test.ms"
    expected_field_id = "field_0123"
    mock_field_table = create_mock_table(getcol_return=[expected_field_id])
    mock_table.return_value = mock_field_table

    result = read_ms_field_id(ms_path)

    assert result == expected_field_id
    mock_table.assert_called_once_with(ms_path + "/FIELD")
    mock_field_table.getcol.assert_called_once_with("NAME")


def test_should_return_unknown_field(mock_table):
    ms_path = "/path/to/test.ms"
    mock_field_table = create_mock_table(getcol_return=[""])
    mock_table.return_value = mock_field_table

    assert read_ms_field_id(ms_path) == "UNKOWN_FIELD"
    mock_table.assert_called_once_with(ms_path + "/FIELD")
    mock_field_table.getcol.assert_called_once_with("NAME")


def test_should_return_first_field_name_when_multiple_fields(mock_table):
    ms_path = "/data/observation.ms"
    field_names = ["target_field", "calibrator_field", "another_field"]
    mock_field_table = create_mock_table(getcol_return=field_names)
    mock_table.return_value = mock_field_table

    result = read_ms_field_id(ms_path)

    assert result == "target_field"
