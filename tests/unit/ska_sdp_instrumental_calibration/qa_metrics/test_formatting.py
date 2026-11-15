from ska_sdp_instrumental_calibration.qa_metrics.formatting import (
    format_flagging_statistics,
    format_metadata,
    format_station_name,
    mjd_seconds_to_utc,
)


def test_should_convert_mjd_seconds_to_utc():
    # MJD 60000 is 2023-02-25
    assert mjd_seconds_to_utc(60000 * 86400.0) == "2023-02-25 00:00:00 UTC"


def test_should_format_station_name():
    assert format_station_name("s0000 (station345_S8-1)") == "S8-1"
    assert format_station_name("station_a") == "station_a"


def test_should_format_metadata():
    metadata = {
        "source_name": "source_name",
        "observation_parameters": {
            "start_time": 60000 * 86400.0,
            "channels": 2,
        },
        "station_configuration": {"0": "s0000 (station345_S8-1)"},
    }

    assert format_metadata(metadata) == {
        "source_name": "source_name",
        "observation_parameters": {
            "start_time": "2023-02-25 00:00:00 UTC",
            "channels": 2,
        },
        "station_configuration": {"0": "S8-1"},
    }


def test_should_format_flagging_statistics():
    rows = [{"station_id": "0", "station_name": "s0000 (station345_S8-1)"}]

    assert format_flagging_statistics(rows) == [
        {"station_id": "0", "station_name": "S8-1"}
    ]
