import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.schema.calibration_set import (  # noqa: E501
    GainCalibrationSetXds,
)
from ska_sdp_instrumental_calibration.qa_metrics.domain import (
    FlaggingStatistics,
)
from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.repository.field_metadata import (  # noqa: E501
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.service.flagging_statistics import (  # noqa: E501
    get_final_gaintable,
    get_flagged_from_gaintable,
)


def _field_metadata(sections):
    return FieldMetadata("field_a", "/qa/dir", sections=sections)


def test_should_get_final_gaintable_of_a_field():
    field_metadata = _field_metadata(
        {"metadata_summary": {"final_gaintable": "/qa/dir/field_a.h5parm"}}
    )

    assert get_final_gaintable(field_metadata) == "/qa/dir/field_a.h5parm"


@pytest.mark.parametrize(
    "sections",
    [
        {},
        {"metadata_summary": {}},
        {"metadata_summary": {"final_gaintable": ""}},
    ],
)
def test_should_raise_if_no_final_gaintable_is_recorded(sections):
    with pytest.raises(
        RunMetadataError, match="No final gaintable recorded for field field_a"
    ):
        get_final_gaintable(_field_metadata(sections))


POLARIZATIONS = ["XX", "XY", "YX", "YY"]


def _gaintable(weight):
    """
    Gaintable of 2 stations with the given weights of shape (time,
    antenna, frequency, polarization)
    """
    ntimes, _, nchannels, _ = weight.shape
    return GainCalibrationSetXds.constructor(
        gain=np.ones(weight.shape, dtype=complex),
        weight=weight,
        residual=np.zeros((ntimes, nchannels, len(POLARIZATIONS))),
        time=np.arange(ntimes, dtype=float),
        interval=np.full(ntimes, np.nan),
        frequency=np.linspace(1e8, 2e8, nchannels),
        antenna_name=np.array(["station_a", "station_b"]),
        polarization=np.array(POLARIZATIONS),
        jones_type="B",
    )


def test_should_get_flagged_percentage_after_from_parallel_hand_weights():
    # 2 times, 2 stations, 2 channels: 8 parallel hand gains per station
    weight = np.ones((2, 2, 2, 4))
    # station_a: 1 of its 8 parallel hand gains is flagged (XX, t0, c0)
    weight[0, 0, 0, 0] = 0
    # Cross hands (XY, YX) are not counted
    weight[:, 1, :, 1:3] = 0
    flagged_before = {"station_a": 0.0, "station_b": 10.0}

    statistics = get_flagged_from_gaintable(_gaintable(weight), flagged_before)

    assert statistics == FlaggingStatistics(
        flagged_before=flagged_before,
        flagged_after={"station_a": 12.5, "station_b": 0.0},
    )
