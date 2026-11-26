import tarfile
from pathlib import Path

import dask
import numpy as np
import pytest
from casacore.tables import table
from xradio.measurement_set import convert_msv2_to_processing_set

from ska_sdp_instrumental_calibration.data_managers.processing_set import (
    _concat_along_time,
    _resolve_single,
    _select_field_times,
    chunk_ms_xdt,
    detect_input_type,
    get_array_location,
    get_baseline_antenna_indices,
    get_field_name,
    get_phase_centre,
    get_uvw_for_prediction,
    input_prefix,
    is_msv2,
    is_processing_set,
    msv4_time_to_mjd_seconds,
    open_msv4_selection,
    read_antenna_response_data_from_msv4,
    read_msv2_selection_shape,
)
from ska_sdp_instrumental_calibration.data_managers.telescope import Telescope
from ska_sdp_instrumental_calibration.data_managers.visibility import (
    read_antenna_response_data,
    read_visibility_from_zarr,
    write_ms_to_zarr,
)

TEST_ROOT = Path(__file__).resolve().parent / "../../../"
MS_TAR = TEST_ROOT / "e2e/resources/test.ms.tgz"

# Shape of the test MS: (time, baseline, frequency, polarization)
MS_SHAPE = (20, 153, 100, 4)


@pytest.fixture(scope="module")
def ms_path(tmp_path_factory):
    temp_dir: Path = tmp_path_factory.mktemp("msv2")
    with tarfile.open(MS_TAR, "r:*") as tar:
        tar.extractall(path=temp_dir)
        ms_name = tar.getnames()[0].split("/")[0]
    return (temp_dir / ms_name).as_posix()


@pytest.fixture(scope="module")
def ps_path(ms_path, tmp_path_factory):
    path = (tmp_path_factory.mktemp("ps") / "test.ps.zarr").as_posix()
    # Stored with time chunks, unlike Processing Sets converted by
    # load_data, so that chunk_ms_xdt has to rechunk
    convert_msv2_to_processing_set(
        in_file=ms_path,
        out_file=path,
        partition_scheme=["FIELD_ID"],
        main_chunksize={"time": 5},
        persistence_mode="w",
    )
    return path


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/obs1.ms", "obs1"),
        ("/data/obs1.ms/", "obs1"),
        ("/data/obs1.ps.zarr", "obs1"),
        ("/data/obs1.zarr", "obs1"),
        ("/data/obs1.ps", "obs1"),
        ("/data/obs1", "obs1"),
        ("/data/.ms", ".ms"),
    ],
)
def test_input_prefix_should_strip_known_suffixes(path, expected):
    assert input_prefix(path) == expected


def test_should_detect_msv2_and_processing_sets(ms_path, ps_path):
    assert is_msv2(ms_path)
    assert not is_processing_set(ms_path)
    assert is_processing_set(ps_path)
    assert not is_msv2(ps_path)

    assert detect_input_type([ms_path]) == "msv2"
    assert detect_input_type([ps_path]) == "ps"


def test_should_raise_for_mixed_inputs(ms_path, ps_path):
    with pytest.raises(ValueError, match="mix MSv2"):
        detect_input_type([ms_path, ps_path])


def test_should_raise_for_unknown_or_missing_inputs(tmp_path):
    with pytest.raises(ValueError, match="neither an MSv2"):
        detect_input_type([tmp_path.as_posix()])

    with pytest.raises(ValueError, match="No input"):
        detect_input_type([])


def test_should_read_msv2_selection_shape(ms_path):
    assert read_msv2_selection_shape(ms_path, 0, 0) == MS_SHAPE


def test_should_raise_reading_shape_for_absent_selection(ms_path):
    with pytest.raises(ValueError, match="DATA_DESC_ID 5 not found"):
        read_msv2_selection_shape(ms_path, 0, 5)

    with pytest.raises(ValueError, match="No rows with FIELD_ID 3"):
        read_msv2_selection_shape(ms_path, 3, 0)


def test_should_open_the_only_field_and_spw(ps_path):
    ms_xdt = open_msv4_selection(ps_path)

    assert get_field_name(ms_xdt) == "CAL_FIELD_0"
    assert ms_xdt.ds.frequency.attrs["spectral_window_name"] == "spw_0"
    assert ms_xdt.ds.VISIBILITY.shape == MS_SHAPE
    assert list(ms_xdt.attrs["data_groups"]) == ["base"]
    # Variables of other data groups are dropped
    assert "VISIBILITY_MODEL" not in ms_xdt.ds
    assert "antenna_xds" in ms_xdt.children
    assert "phased_array_xds" in ms_xdt.children


def test_should_select_data_group_field_and_spw_by_name(ps_path):
    ms_xdt = open_msv4_selection(
        ps_path,
        field_name="CAL_FIELD_0",
        spw_name="spw_0",
        scan_intents=["CALIBRATE_BANDPASS#ON_SOURCE"],
        data_group="model",
    )

    assert "VISIBILITY_MODEL" in ms_xdt.ds
    assert "VISIBILITY" not in ms_xdt.ds
    assert list(ms_xdt.attrs["data_groups"]) == ["model"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"field_name": "OTHER"}, "field_name 'OTHER' not found"),
        ({"spw_name": "spw_9"}, "spw_name 'spw_9' not found"),
        ({"data_group": "corrected"}, "Data group 'corrected' not found"),
        ({"scan_intents": ["OBSERVE_TARGET#ON_SOURCE"]}, "No MSv4 data"),
    ],
)
def test_should_raise_for_absent_selection(ps_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        open_msv4_selection(ps_path, **kwargs)


def test_resolve_single_should_require_a_name_when_ambiguous():
    assert _resolve_single("field", None, {"A"}, "ps") == "A"
    assert _resolve_single("field", "B", {"A", "B"}, "ps") == "B"

    with pytest.raises(ValueError, match=r"2 fields \['A', 'B'\]"):
        _resolve_single("field", None, {"A", "B"}, "ps")


def test_should_keep_only_time_steps_of_the_selected_field(ps_path):
    ms_xdt = open_msv4_selection(ps_path)
    field_names = np.where(
        np.arange(MS_SHAPE[0]) % 2 == 0, "CAL_FIELD_0", "OTHER_1"
    )
    two_fields = ms_xdt.copy(deep=False)
    two_fields.ds = ms_xdt.to_dataset().assign_coords(
        field_name=("time", field_names)
    )

    selected = _select_field_times(two_fields, "CAL_FIELD_0")

    np.testing.assert_array_equal(
        selected.ds.time.values, ms_xdt.ds.time.values[::2]
    )
    assert set(selected.ds.field_name.values) == {"CAL_FIELD_0"}
    assert _select_field_times(ms_xdt, "CAL_FIELD_0") is ms_xdt


def test_should_concat_nodes_along_time(ps_path):
    ms_xdt = open_msv4_selection(ps_path)
    ds = ms_xdt.to_dataset()
    halves = []
    # Pass the later half first, to check the result is sorted by time
    for time_slice in (slice(10, None), slice(None, 10)):
        node = ms_xdt.copy(deep=False)
        node.ds = ds.isel(time=time_slice)
        halves.append(node)

    combined = _concat_along_time(halves)

    np.testing.assert_array_equal(combined.ds.time.values, ds.time.values)
    np.testing.assert_array_equal(
        combined.ds.VISIBILITY.values, ds.VISIBILITY.values
    )
    assert "antenna_xds" in combined.children


def test_should_raise_concatenating_nodes_with_different_frequencies(
    ps_path,
):
    ms_xdt = open_msv4_selection(ps_path)
    shifted = ms_xdt.copy(deep=False)
    shifted.ds = ms_xdt.to_dataset().assign_coords(
        frequency=ms_xdt.ds.frequency.values + 1.0
    )

    with pytest.raises(ValueError, match="their frequency differ"):
        _concat_along_time([ms_xdt, shifted])


def test_should_chunk_along_frequency_only(ps_path):
    ms_xdt = open_msv4_selection(ps_path)

    chunked = chunk_ms_xdt(ms_xdt, 30)

    assert chunked.ds.chunksizes == {
        "time": (20,),
        "baseline_id": (153,),
        "frequency": (30, 30, 30, 10),
        "polarization": (4,),
        "uvw_label": (3,),
    }
    # The original node is left untouched
    assert ms_xdt.ds.VISIBILITY.chunks != chunked.ds.VISIBILITY.chunks


# The MSv4 helpers below replace attributes of the MSv2 Visibility. They are
# checked against the MSv2 Visibility reader on the same MS.


@pytest.fixture(scope="module")
def msv2_vis(ms_path, tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("vis_cache").as_posix()
    chunks = {
        "time": -1,
        "baselineid": -1,
        "frequency": -1,
        "polarisation": -1,
        "spatial": -1,
    }
    dask.compute(write_ms_to_zarr(ms_path, cache_dir, chunks))
    return read_visibility_from_zarr(cache_dir, chunks)


@pytest.fixture(scope="module")
def ms_xdt(ps_path):
    return open_msv4_selection(ps_path)


def _msv2_baseline_index(msv2_vis, antenna1, antenna2):
    """Position in the MSv2 Visibility of each (antenna1, antenna2)."""
    pairs = list(
        zip(msv2_vis.antenna1.values, msv2_vis.antenna2.values, strict=True)
    )
    return np.array(
        [pairs.index(pair) for pair in zip(antenna1, antenna2, strict=True)]
    )


def test_should_convert_msv4_time_to_msv2_time(ms_path, ms_xdt):
    with table(ms_path, ack=False) as tb:
        msv2_times = np.unique(tb.getcol("TIME"))

    np.testing.assert_array_equal(
        msv4_time_to_mjd_seconds(ms_xdt.ds.time.values), msv2_times
    )


def test_should_read_phase_centre_and_location_as_msv2_reader(
    ms_xdt, msv2_vis
):
    phase_centre = get_phase_centre(ms_xdt)
    location = get_array_location(ms_xdt)

    assert phase_centre.frame.name == msv2_vis.phasecentre.frame.name
    assert phase_centre.ra.rad == msv2_vis.phasecentre.ra.rad
    assert phase_centre.dec.rad == msv2_vis.phasecentre.dec.rad
    np.testing.assert_array_equal(
        [c.to_value("m") for c in location.geocentric],
        [c.to_value("m") for c in msv2_vis.configuration.location.geocentric],
    )


def test_should_read_baseline_antennas_and_uvw_as_msv2_reader(
    ms_xdt, msv2_vis
):
    antenna1, antenna2 = get_baseline_antenna_indices(ms_xdt)
    np.testing.assert_array_equal(
        ms_xdt["antenna_xds"].ds.antenna_name.values,
        msv2_vis.configuration.names.values,
    )
    baseline_index = _msv2_baseline_index(msv2_vis, antenna1, antenna2)

    uvw = get_uvw_for_prediction(ms_xdt)

    assert uvw.dims == ("time", "baseline_id", "uvw_label")
    # MSv4 keeps the MSv2 UVW sign, the Visibility reader negates it
    np.testing.assert_array_equal(
        uvw.values, msv2_vis.uvw.values[:, baseline_index, :]
    )
    np.testing.assert_array_equal(uvw.values, -ms_xdt.ds.UVW.values)


def test_should_read_antenna_response_data_as_msv2_reader(ms_path, ms_xdt):
    expected = read_antenna_response_data(ms_path)

    actual = read_antenna_response_data_from_msv4(ms_xdt)

    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert isinstance(value, np.ndarray), key
        np.testing.assert_array_equal(value, expected[key], err_msg=key)


def test_should_raise_without_phased_array(ms_xdt):
    without_phased_array = ms_xdt.copy(deep=False)
    del without_phased_array["phased_array_xds"]

    with pytest.raises(ValueError, match="no phased_array_xds"):
        read_antenna_response_data_from_msv4(without_phased_array)


def test_should_compute_same_beams_from_msv4_and_msv2(ms_path, ms_xdt):
    frequencies = ms_xdt.ds.frequency.values[:3]
    time = msv4_time_to_mjd_seconds(ms_xdt.ds.time.values[0])
    direction = np.array([0.0, 0.0, 1.0])

    msv2_telescope = Telescope(ms_path, "oskar_dipole_cos")
    msv4_telescope = Telescope.from_msv4(ms_xdt, "oskar_dipole_cos")

    np.testing.assert_array_equal(
        msv4_telescope.station_response(
            time, frequencies, direction, direction
        ),
        msv2_telescope.station_response(
            time, frequencies, direction, direction
        ),
    )


@pytest.mark.parametrize("eb_ms", ["ms_path", "ps_path"])
def test_should_read_same_response_data_from_eb_ms(request, ms_xdt, eb_ms):
    expected = Telescope.for_visibilities(ms_xdt, "oskar_dipole_cos")

    actual = Telescope.for_visibilities(
        ms_xdt, "oskar_dipole_cos", request.getfixturevalue(eb_ms)
    )

    assert actual.response_data.keys() == expected.response_data.keys()
    for key, value in actual.response_data.items():
        np.testing.assert_array_equal(
            value, expected.response_data[key], err_msg=key
        )
