import numpy as np
import xarray as xr

from ska_sdp_instrumental_calibration.xarray_processors._utils import (
    with_chunks,
)


def test_should_chunk_xarray_object_with_valid_chunks():
    data = xr.DataArray(np.arange(12).reshape(4, 3), dims=["a", "b"])

    chunks = {"a": 2, "c": 4}

    new_data = with_chunks(data, chunks)

    assert dict(new_data.chunksizes) == {"a": (2, 2), "b": (3,)}
