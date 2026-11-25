"""
Flagging statistics of a field: the products of the flagging statistics
section, looked up in the run metadata of the field.
"""

from ...data_managers.schema.calibration_set import GainCalibrationSetXds
from ..domain import FlaggingStatistics
from ..exceptions import RunMetadataError
from ..repository.field_metadata import FieldMetadata
from ..repository.manifest_schema import ManifestSchema


def get_final_gaintable(field_metadata: FieldMetadata) -> str:
    """
    Get the final gaintable path of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field.

    Returns
    -------
    str
        Path of the final gaintable of the field.

    Raises
    ------
    RunMetadataError
        If no final gaintable is recorded for the field.
    """
    gaintable = field_metadata.get_data_for(
        ManifestSchema.METADATA_SUMMARY.FINAL_GAINTABLE
    )
    if not gaintable:
        raise RunMetadataError(
            f"No final gaintable recorded for field "
            f"{field_metadata.field_name}"
        )
    return gaintable


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
