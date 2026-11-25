import numpy as np

from ska_sdp_instrumental_calibration.qa_metrics.view.qa_metric_report import (
    template_env,
)
from ska_sdp_instrumental_calibration.qa_metrics.view.tables import (
    CABLE_DELAYS_COLUMNS,
    get_bandpass_fit_metrics_columns,
)

BANDPASS_COLUMNS = get_bandpass_fit_metrics_columns(0.8, 0.75)


def _render_table(columns, rows):
    """HTML of the data_table macro of components.html"""
    return template_env.from_string(
        '{% from "components.html" import data_table %}'
        '{{ data_table(columns, rows, "stations") }}'
    ).render(columns=columns, rows=rows)


def test_should_set_thresholds_on_bandpass_fit_columns():
    phase, gain = BANDPASS_COLUMNS[1], BANDPASS_COLUMNS[2]

    assert [column.threshold for column in phase.columns] == [0.8, 0.8]
    assert [column.threshold for column in gain.columns] == [0.75, 0.75]


def test_should_show_acceptable_threshold_in_group_header():
    html = _render_table(BANDPASS_COLUMNS, [])

    assert "Acceptable threshold: <span>0.8</span>" in html
    assert "Acceptable threshold: <span>0.75</span>" in html


def test_should_mark_values_below_threshold_and_nan():
    html = _render_table(
        BANDPASS_COLUMNS,
        [
            {
                "station": "s0000",
                "x_phase_fit_quality": 0.5,
                "y_phase_fit_quality": 0.9,
                "x_gains_fit_quality": np.nan,
                "y_gains_fit_quality": 0.75,
            }
        ],
    )

    # Only 0.5 is below its threshold: 0.75 is equal to it
    assert html.count("below-threshold") == 1
    assert html.count("not-a-number") == 1


def test_should_not_mark_columns_without_threshold():
    html = _render_table(
        CABLE_DELAYS_COLUMNS,
        [{"station": "s0000", "x_delay": -1.0, "y_delay": np.nan}],
    )

    assert "below-threshold" not in html
    assert "not-a-number" not in html
    assert "Acceptable threshold" not in html
