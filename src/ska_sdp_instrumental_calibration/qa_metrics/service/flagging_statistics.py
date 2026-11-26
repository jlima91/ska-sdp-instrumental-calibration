"""
Flagging statistics of a field: the flagged percentage of each station
after calibration, from its final gaintable.
"""

from ...data_managers.schema.calibration_set import GainCalibrationSetXds
from ..domain.flagging_statistics import FlaggingStatistics


def get_flagged_from_gaintable(
    gaintable: GainCalibrationSetXds, flagged_before: dict[str, float]
) -> FlaggingStatistics:
    """
    Get the flagging statistics of a field from its final gaintable.

    A gain is flagged where its weight is 0. Only the parallel hand
    gains (XX, YY) solved by bandpass are counted.

    Parameters
    ----------
    gaintable : GainCalibrationSetXds
        Final gaintable of the field, with ``CALPARAM_WEIGHT`` of
        dimensions (time, antenna_name, frequency, polarization).
    flagged_before : dict[str, float]
        Percentage of gains that cannot be solved before calibration,
        keyed by MS station name.

    Returns
    -------
    FlaggingStatistics
        Flagging statistics of the field.
    """
    is_flagged = gaintable.CALPARAM_WEIGHT.sel(polarization=["XX", "YY"]) == 0
    # Percentage of the gains of each station: all its times,
    # channels and parallel hands
    flagged_percentage = (
        is_flagged.mean(dim=("time", "frequency", "polarization")) * 100
    )

    return FlaggingStatistics(
        flagged_before=flagged_before,
        flagged_after={
            str(name): float(percentage)
            for name, percentage in zip(
                gaintable.antenna_name.values, flagged_percentage.values
            )
        },
    )
