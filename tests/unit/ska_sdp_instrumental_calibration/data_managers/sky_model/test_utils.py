from mock import call, patch

from ska_sdp_instrumental_calibration.data_managers.sky_model import utils


@patch("builtins.open")
def test_write_csv_writes_rows_correctly(mock_open):
    mock_open.return_value = mock_open
    mock_open.__enter__.return_value = mock_open

    filepath = "dummy.csv"
    rows = [["a", "b", "c"], ["1", "2", "3"], ["x", "y", "z"]]

    utils.write_csv(filepath, rows)

    mock_open.assert_called_once_with(filepath, "w", encoding="utf-8")

    expected_calls = [
        call("a,b,c\n"),
        call("1,2,3\n"),
        call("x,y,z\n"),
    ]

    mock_open.write.assert_has_calls(expected_calls)
