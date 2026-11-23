import json

import numpy as np
import pytest
from mock import patch

from ska_sdp_instrumental_calibration.data_managers.data_export import (
    export_clock_to_h5parm,
)
from ska_sdp_instrumental_calibration.data_managers.schema.calibration_set import (  # noqa: E501
    GainCalibrationSetXds,
)
from ska_sdp_instrumental_calibration.qa_metrics.domain.qa_metadata import (
    FileInformation,
    ObservationParameters,
    StationConfiguration,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.field_sections import (
    section_builder,
)
from ska_sdp_instrumental_calibration.qa_metrics.products import gaintable
from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.sections import (
    CableDelaysSection,
    ErrorSection,
    FlaggingStatisticsSection,
    MetadataSection,
)
from ska_sdp_instrumental_calibration.qa_metrics.tables import (
    CABLE_DELAYS_COLUMNS,
    FLAGGING_STATISTICS_COLUMNS,
)
from ska_sdp_instrumental_calibration.xarray_processors.delay import DelayTable

READ_GAINTABLE = (
    "ska_sdp_instrumental_calibration.qa_metrics.field_sections"
    ".read_gaintable"
)
GAINTABLE = "/qa/dir/field_a.h5parm"
POLARIZATIONS = ["XX", "XY", "YX", "YY"]


@pytest.fixture(autouse=True)
def clear_gaintable_cache():
    """
    A gaintable read cached by one test must not leak into the next
    """
    # pylint: disable-next=protected-access
    gaintable._read_one.cache_clear()
    yield
    # pylint: disable-next=protected-access
    gaintable._read_one.cache_clear()


def _gaintable(station_names, weight=None, nchannels=2):
    """
    Final gaintable of one time, with unit weights unless given, of shape
    (time, antenna, frequency, polarization)
    """
    shape = (1, len(station_names), nchannels, len(POLARIZATIONS))
    return GainCalibrationSetXds.constructor(
        gain=np.ones(shape, dtype=complex),
        weight=np.ones(shape) if weight is None else weight,
        residual=np.zeros((1, nchannels, len(POLARIZATIONS))),
        time=np.array([10.0]),
        interval=np.array([np.nan]),
        frequency=np.linspace(1e8, 2e8, nchannels),
        antenna_name=np.array(station_names),
        polarization=np.array(POLARIZATIONS),
        jones_type="B",
    )


def _field_metadata(metadata_summary):
    return FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={"metadata_summary": metadata_summary},
    )


@patch(READ_GAINTABLE)
def test_should_get_metadata_section_of_a_field(read_gaintable_mock):
    read_gaintable_mock.return_value = _gaintable(["station_a", "station_b"])
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0, 30.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    assert section_builder.build(
        MetadataSection, field_metadata
    ) == MetadataSection(
        file_information=FileInformation(
            input_ms=["/data/a.ms"], output_path="/qa/dir"
        ),
        observation_parameters=ObservationParameters(
            start_time=5.0,
            observation_length=30.0,
            integration_time=10.0,
            channels=64,
        ),
        station_configuration=StationConfiguration(
            {"0": "station_a", "1": "station_b"}
        ),
    )
    read_gaintable_mock.assert_called_once_with([GAINTABLE])


@patch(READ_GAINTABLE)
def test_should_join_observation_parameters_of_all_ms_along_time(
    read_gaintable_mock,
):
    read_gaintable_mock.return_value = _gaintable(["station_a"])
    # MSs are not recorded in time order
    field_metadata = _field_metadata(
        {
            "MS": ["/data/b.ms", "/data/a.ms"],
            "timestamps": [[40.0, 50.0], [10.0, 20.0, 30.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    qa_metadata = section_builder.build(MetadataSection, field_metadata)

    assert qa_metadata.observation_parameters == ObservationParameters(
        start_time=5.0,
        observation_length=50.0,
        integration_time=10.0,
        channels=64,
    )
    assert qa_metadata.file_information.input_ms == [
        "/data/b.ms",
        "/data/a.ms",
    ]


@patch(READ_GAINTABLE)
def test_should_leave_channels_unset_if_not_recorded(read_gaintable_mock):
    # Missing channels must not stop the report; it shows them as unknown
    read_gaintable_mock.return_value = _gaintable(["station_a"])
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0, 30.0]],
            "final_gaintable": GAINTABLE,
        }
    )

    qa_metadata = section_builder.build(MetadataSection, field_metadata)

    assert qa_metadata.observation_parameters.channels is None


def _assert_metadata_error_section(section, cause_match):
    """
    The section is the error section of the metadata summary of field_a,
    caused by a RunMetadataError matching cause_match
    """
    assert isinstance(section, ErrorSection)
    assert section.title == MetadataSection.title
    assert "field_a" in str(section.error)
    assert isinstance(section.error.__cause__, RunMetadataError)
    assert cause_match in str(section.error.__cause__)


@pytest.mark.parametrize(
    "metadata_summary",
    [
        # Each case misses exactly one of the required labels; channels
        # is optional, so it is recorded in every case
        {
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        },
        {"MS": ["/data/a.ms"], "final_gaintable": GAINTABLE, "channels": 64},
        {"MS": ["/data/a.ms"], "timestamps": [[10.0, 20.0]], "channels": 64},
        {
            "MS": [],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        },
        {},
    ],
)
@patch(READ_GAINTABLE)
def test_should_get_error_section_if_qa_metadata_is_missing(
    read_gaintable_mock, metadata_summary
):
    field_metadata = _field_metadata(metadata_summary)

    section = section_builder.build(MetadataSection, field_metadata)

    _assert_metadata_error_section(section, cause_match="field_a")
    read_gaintable_mock.assert_not_called()


@patch(READ_GAINTABLE)
def test_should_get_error_section_if_final_gaintable_cannot_be_read(
    read_gaintable_mock,
):
    # Reader errors (not produced, corrupt, no axes) are covered by the
    # reader tests: the section only has to show them
    read_gaintable_mock.side_effect = RunMetadataError(
        f"Gaintable {GAINTABLE} was not produced"
    )
    field_metadata = _field_metadata(
        {
            "MS": ["/data/a.ms"],
            "timestamps": [[10.0, 20.0]],
            "final_gaintable": GAINTABLE,
            "channels": 64,
        }
    )

    section = section_builder.build(MetadataSection, field_metadata)

    _assert_metadata_error_section(section, cause_match="was not produced")


def _flagging_field_metadata(pre_flagging):
    return FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "metadata_summary": {"final_gaintable": GAINTABLE},
            "flagging": {"pre_flagging": pre_flagging},
        },
    )


def _write_pre_flagging(path, unsolvable, nchannels):
    nstations = len(unsolvable)
    path.write_text(
        json.dumps(
            {
                "flagged": [0] * nstations,
                "total": [1] * nstations,
                "unsolvable": unsolvable,
                "nchannels": nchannels,
            }
        )
    )
    return str(path)


@patch(READ_GAINTABLE)
def test_should_get_flagging_statistics_section_of_a_field(
    read_gaintable_mock, tmp_path
):
    # (time, antenna, frequency, polarization): 2 stations, 2 channels
    weight = np.ones((1, 2, 2, 4))
    # station_a: 1 of its 4 parallel hand gains flagged after calibration
    weight[0, 0, 0, 0] = 0
    read_gaintable_mock.return_value = _gaintable(
        ["station_a", "station_b"], weight=weight
    )
    # station_b: 1 of its 2 channels cannot be solved before calibration
    field_metadata = _flagging_field_metadata(
        {
            "ms_a": _write_pre_flagging(
                tmp_path / "a.json", unsolvable=[0, 1], nchannels=2
            )
        }
    )

    section = section_builder.build(FlaggingStatisticsSection, field_metadata)

    assert section == FlaggingStatisticsSection(
        columns=FLAGGING_STATISTICS_COLUMNS,
        rows=[
            {
                "station_id": "0",
                "station_name": "station_a",
                "flagged_before": 0.0,
                "flagged_after": 25.0,
                "flagged_additional": 25.0,
            },
            {
                "station_id": "1",
                "station_name": "station_b",
                "flagged_before": 50.0,
                "flagged_after": 0.0,
                "flagged_additional": -50.0,
            },
        ],
        totals={"flagged_before": 25.0, "flagged_after": 12.5},
    )
    # Stations and weights come from a single read of the gaintable
    read_gaintable_mock.assert_called_once_with([GAINTABLE])


@patch(READ_GAINTABLE)
def test_should_get_error_section_if_pre_flagging_is_missing(
    read_gaintable_mock, tmp_path
):
    read_gaintable_mock.return_value = _gaintable(["station_a"])
    field_metadata = _flagging_field_metadata(
        {"ms_a": str(tmp_path / "missing.json")}
    )

    section = section_builder.build(FlaggingStatisticsSection, field_metadata)

    assert isinstance(section, ErrorSection)
    assert section.title == FlaggingStatisticsSection.title
    assert "field_a" in str(section.error)
    assert "was not produced" in str(section.error.__cause__)


@patch(READ_GAINTABLE)
def test_should_get_error_section_if_final_gaintable_is_not_recorded(
    read_gaintable_mock,
):
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    section = section_builder.build(FlaggingStatisticsSection, field_metadata)

    assert isinstance(section, ErrorSection)
    assert "No final gaintable" in str(section.error.__cause__)
    read_gaintable_mock.assert_not_called()


@pytest.mark.parametrize(
    "section_type", [MetadataSection, FlaggingStatisticsSection]
)
def test_should_get_error_section_if_final_gaintable_is_corrupt(
    section_type, tmp_path
):
    # End to end through the reader: a corrupt gaintable is a bad
    # product, shown in the section rather than failing the report
    corrupt = tmp_path / "corrupt.h5parm"
    corrupt.write_text("not an HDF5 file")
    field_metadata = FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "metadata_summary": {
                "MS": ["/data/a.ms"],
                "timestamps": [[10.0, 20.0]],
                "final_gaintable": str(corrupt),
            },
            "flagging": {
                "pre_flagging": {
                    "ms_a": _write_pre_flagging(
                        tmp_path / "a.json", unsolvable=[0], nchannels=1
                    )
                }
            },
        },
    )

    section = section_builder.build(section_type, field_metadata)

    assert isinstance(section, ErrorSection)
    assert section.title == section_type.title
    assert "cannot be read" in str(section.error.__cause__)


def _write_delaytable(path, time, x_delay, y_delay):
    """
    Delay table of one MS with the same X and Y delay, in seconds, for
    station_a and twice it for station_b
    """
    delay = np.array([[[x_delay, y_delay], [2 * x_delay, 2 * y_delay]]])
    export_clock_to_h5parm(
        DelayTable.constructor(
            delay=delay,
            offset=np.zeros_like(delay),
            time=np.array([time]),
            antenna_name=np.array(["station_a", "station_b"]),
            pol=["XX", "YY"],
        ),
        str(path),
    )
    return str(path)


def test_should_get_cable_delays_section_of_a_field(tmp_path):
    # Three MSs: their delay tables are joined along time and each station
    # gets its median delay, unmoved by the badly solved scan of ms_c
    field_metadata = FieldMetadata(
        "field_a",
        "/qa/dir",
        sections={
            "delay": {
                "delaytable": {
                    "ms_b": _write_delaytable(
                        tmp_path / "b.h5parm", 200.0, 20e-9, 40e-9
                    ),
                    "ms_a": _write_delaytable(
                        tmp_path / "a.h5parm", 100.0, 10e-9, 30e-9
                    ),
                    "ms_c": _write_delaytable(
                        tmp_path / "c.h5parm", 300.0, 100e-9, 300e-9
                    ),
                }
            }
        },
    )

    section = section_builder.build(CableDelaysSection, field_metadata)

    assert section == CableDelaysSection(
        columns=CABLE_DELAYS_COLUMNS,
        rows=[
            {
                "station_id": "0",
                "station_name": "station_a",
                "x_delay": pytest.approx(20.0),
                "y_delay": pytest.approx(40.0),
            },
            {
                "station_id": "1",
                "station_name": "station_b",
                "x_delay": pytest.approx(40.0),
                "y_delay": pytest.approx(80.0),
            },
        ],
    )


def test_should_get_error_section_if_no_delaytable_is_recorded():
    field_metadata = FieldMetadata("field_a", "/qa/dir", sections={})

    section = section_builder.build(CableDelaysSection, field_metadata)

    assert isinstance(section, ErrorSection)
    assert section.title == CableDelaysSection.title
    assert "field_a" in str(section.error)
    assert "No delay table recorded" in str(section.error.__cause__)
