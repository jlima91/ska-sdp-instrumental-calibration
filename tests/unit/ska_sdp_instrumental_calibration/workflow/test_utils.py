import numpy as np
import xarray as xr
from mock import Mock

from ska_sdp_instrumental_calibration.workflow.utils import with_chunks

plot_gaintable = Mock(name="plot_gaintable")
plot_all_stations = Mock(name="plot_all_stations")
subplot_gaintable = Mock(name="subplot_gaintable")


def test_should_chunk_xarray_object_with_valid_chunks():
    data = xr.DataArray(np.arange(12).reshape(4, 3), dims=["a", "b"])

    chunks = {"a": 2, "c": 4}

    new_data = with_chunks(data, chunks)

    assert dict(new_data.chunksizes) == {"a": (2, 2), "b": (3,)}
