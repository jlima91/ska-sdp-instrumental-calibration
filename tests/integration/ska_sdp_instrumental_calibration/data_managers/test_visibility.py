import os

import numpy as np

from ska_sdp_instrumental_calibration.data_managers.visibility import (
    check_if_cache_files_exist,
    read_visibility_from_zarr,
    write_ms_to_zarr,
)


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
    # np.testing.assert_allclose((gen_vis.weight.data),(zarred_dataset.weight.data))

    np.testing.assert_allclose(
        (gen_vis.frequency.data), (zarred_dataset.frequency.data)
    )
    np.testing.assert_allclose((gen_vis.time.data), (zarred_dataset.time.data))
