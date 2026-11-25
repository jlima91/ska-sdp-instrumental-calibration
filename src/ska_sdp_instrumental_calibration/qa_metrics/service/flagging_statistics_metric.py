"""
Flagging statistics of a field: the products of the flagging statistics
section, looked up in the run metadata of the field.
"""

from ..exceptions import RunMetadataError
from ..repository.qa_section import QASection
from ..repository.run_metadata import FieldMetadata


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
        QASection.METADATA_SUMMARY.FINAL_GAINTABLE
    )
    if not gaintable:
        raise RunMetadataError(
            f"No final gaintable recorded for field "
            f"{field_metadata.field_name}"
        )
    return gaintable


def get_pre_flagging_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the pre-flagging data files of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one pre-flagging data file
        recorded per MS prefix.

    Returns
    -------
    list[str]
        Pre-flagging data file of each measurement set of the field. Empty
        if none is recorded: the reader raises for it.
    """
    pre_flagging = (
        field_metadata.get_data_for(QASection.FLAGGING.PRE_FLAGGING) or {}
    )
    return list(pre_flagging.values())
