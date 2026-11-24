"""
Cable delays of a field: the products of the cable delays section, looked
up in the run metadata of the field.
"""

from ..qa_section import QASection
from ..run_metadata import FieldMetadata


def get_delaytable_paths(field_metadata: FieldMetadata) -> list[str]:
    """
    Get the delay tables of a field from its run metadata.

    Parameters
    ----------
    field_metadata : FieldMetadata
        Run metadata of the field, with one delay table recorded per MS
        prefix.

    Returns
    -------
    list[str]
        Delay table of each measurement set of the field. Empty if none is
        recorded: the reader raises for it.
    """
    delaytables = field_metadata.get_data_for(QASection.DELAY.DELAYTABLE) or {}
    return list(delaytables.values())
