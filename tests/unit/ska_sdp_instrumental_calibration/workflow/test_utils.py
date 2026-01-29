import numpy as np
from mock import MagicMock, Mock, call, patch

from ska_sdp_instrumental_calibration.workflow.utils import (
    create_clock_soltab_datasets,
    create_soltab_datasets,
    create_soltab_group,
    get_ms_metadata,
    get_phasecentre,
    normalize_data,
    with_chunks,
)

plot_gaintable = Mock(name="plot_gaintable")
plot_all_stations = Mock(name="plot_all_stations")
subplot_gaintable = Mock(name="subplot_gaintable")


@patch("ska_sdp_instrumental_calibration.workflow.utils.table")
@patch("ska_sdp_instrumental_calibration.workflow.utils.SkyCoord")
def test_should_get_phasecentre(sky_coord_mock, table_mock):
    fieldtab_mock = MagicMock(name="fieldtab")
    phase_dir = np.array([[[1, 2]]])
    fieldtab_mock.getcol.return_value = phase_dir
    table_mock.return_value = fieldtab_mock

    phase_center = get_phasecentre("MSNAME")

    table_mock.assert_called_once_with("MSNAME/FIELD", ack=False)
    fieldtab_mock.getcol.assert_called_once_with("PHASE_DIR")

    sky_coord_mock.assert_called_once_with(
        ra=1, dec=2, unit="radian", frame="icrs", equinox="J2000"
    )

    assert phase_center == sky_coord_mock.return_value


@patch("ska_sdp_instrumental_calibration.workflow.utils.table")
@patch("ska_sdp_instrumental_calibration.workflow.utils.PolarisationFrame")
@patch(
    "ska_sdp_instrumental_calibration.workflow.utils.create_visibility_from_ms"
)
def test_should_get_ms_metadata(create_ms_mock, polframe_mock, table_mock):
    spwtab_mock = MagicMock(name="spwtab")
    spwtab_mock.getcol.side_effect = (["CHAN_FREQ"], ["CHAN_WIDTH"])
    table_mock.return_value = spwtab_mock
    tmp_vis = Mock(name="tmp_vis")
    create_ms_mock.return_value = [tmp_vis]

    ms_metadata = get_ms_metadata("MSNAME")

    create_ms_mock.asserrt_called_once_with(
        "MSNAME",
        start_chan=0,
        ack=False,
        datacolumn="DATA",
        end_chan=0,
        selected_sources=None,
        selected_dds=None,
        average_channels=False,
    )

    table_mock.assert_called_once_with("MSNAME/SPECTRAL_WINDOW", ack=False)
    spwtab_mock.getcol.assert_has_calls(
        [call("CHAN_FREQ"), call("CHAN_WIDTH")]
    )
    polframe_mock.assert_called_once_with(tmp_vis._polarisation_frame)

    assert ms_metadata.uvw == tmp_vis.uvw.data
    assert ms_metadata.baselines == tmp_vis.baselines
    assert ms_metadata.time == tmp_vis.time
    np.testing.assert_array_equal(ms_metadata.frequency, ["CHAN_FREQ"])
    np.testing.assert_array_equal(
        ms_metadata.channel_bandwidth, ["CHAN_WIDTH"]
    )
    assert ms_metadata.integration_time == tmp_vis.integration_time
    assert ms_metadata.configuration == tmp_vis.configuration
    assert ms_metadata.phasecentre == tmp_vis.phasecentre
    assert ms_metadata.polarisation_frame == polframe_mock.return_value
    assert ms_metadata.source == "bpcal"
    assert ms_metadata.meta is None


def test_should_create_soltab_dataset():
    soltab = MagicMock(name="soltab")
    gaintable = MagicMock(name="gaintbale")
    data_mock = MagicMock(name="data")
    gaintable.__getitem__.return_value = data_mock
    val_mock = MagicMock(name="val")
    weight_mock = MagicMock(name="weight")
    soltab.create_dataset.side_effect = ("A", "B", val_mock, weight_mock)

    gaintable.gain.sizes = ["A", "B"]

    val, weight = create_soltab_datasets(soltab, gaintable)

    soltab.create_dataset.assert_has_calls(
        [
            call("A", data=data_mock.data),
            call("B", data=data_mock.data),
            call("val", shape=gaintable.gain.shape, dtype=float),
            call("weight", shape=gaintable.gain.shape, dtype=float),
        ]
    )

    gaintable.__getitem__.assert_has_calls([call("A"), call("B")])
    val_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    weight_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    assert val == val_mock
    assert weight == weight_mock


def test_should_create_clock_soltab_dataset():
    soltab = MagicMock(name="soltab")
    gaintable = MagicMock(name="gaintbale")
    data_mock = MagicMock(name="data")
    gaintable.__getitem__.return_value = data_mock
    val_mock = MagicMock(name="val")
    offset_mock = MagicMock(name="offset")
    soltab.create_dataset.side_effect = ("A", "B", val_mock, offset_mock)

    gaintable.delay.sizes = ["A", "B"]

    val, offset = create_clock_soltab_datasets(soltab, gaintable)

    soltab.create_dataset.assert_has_calls(
        [
            call("A", data=data_mock.data),
            call("B", data=data_mock.data),
            call("val", shape=gaintable.delay.shape, dtype=float),
            call("offset", shape=gaintable.delay.shape, dtype=float),
        ]
    )

    gaintable.__getitem__.assert_has_calls([call("A"), call("B")])
    val_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    offset_mock.attrs.__setitem__.assert_called_once_with(
        "AXES", np.bytes_(b"A,B")
    )

    assert val == val_mock
    assert offset == offset_mock


def test_should_create_soltab_group():
    soltab = MagicMock(name="soltab")
    solset = MagicMock(name="solset")
    solset.create_group.return_value = soltab

    soltb = create_soltab_group(solset, "phase")

    solset.create_group.assert_called_once_with("phase000")
    soltab.attrs.__setitem__.assert_called_once_with(
        "TITLE", np.bytes_("phase")
    )

    assert soltb == soltab


def test_should_create_appropriate_chunks():
    data_array = MagicMock(name="dataarray")
    data_array.dims = ["A", "B", "C"]
    chunks = {"A": 1, "C": 2}

    chunked_dataarray = with_chunks(data_array, chunks)

    assert chunked_dataarray == data_array.chunk.return_value

    data_array.chunk.assert_called_once_with({"A": 1, "C": 2})


def test_should_not_chunk():
    data_array = MagicMock(name="dataarray")
    data_array.dims = ["AA", "BB", "CC"]
    chunks = {"A": 1, "C": 2}

    chunked_dataarray = with_chunks(data_array, chunks)

    assert chunked_dataarray == data_array

    data_array.chunk.assert_not_called()


def test_should_normalize_data():
    data = np.array([1, 2, 3, 4])
    norm_data = normalize_data(data)

    np.testing.assert_allclose(norm_data, data / 10)
