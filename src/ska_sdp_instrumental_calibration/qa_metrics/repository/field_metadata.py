import os
from typing import Any

from .manifest_schema import JSONSerialisable, QASectionLabel


class FieldMetadata:
    """
    Data products recorded for one field.

    Each field gets its own QA report, and a ``FieldMetadata`` holds
    everything needed to generate it, including the QA directory.

    Parameters
    ----------
    field_name : str
        Field name.
    qa_dir : str
        QA directory of the run.
    sections : dict[str, dict[str, JSONSerialisable]], optional
        Data products keyed by QA section, then by label. Empty if not
        given.
    """

    def __init__(
        self,
        field_name,
        qa_dir,
        sections: dict[str, dict[str, JSONSerialisable]] | None = None,
    ):
        self.field_name = field_name
        # str so the field metadata stays JSON serialisable
        self.qa_dir = os.fspath(qa_dir)
        self.sections = {} if sections is None else sections

    def add_data_for(
        self,
        label: QASectionLabel,
        data: JSONSerialisable,
        prefix: str | None = None,
    ):
        """
        Add data to a label. The label validates the data and decides
        whether it overwrites or appends.

        Parameters
        ----------
        label : QASectionLabel
            Label of the data, e.g. ``ManifestSchema.METADATA_SUMMARY.MS``.
        data : JSONSerialisable
            Data of the schema of the label.
        prefix : str, optional
            MS prefix of the data, for labels stored per MS.
        """
        label.store(
            self.sections.setdefault(label.section, {}), data, prefix=prefix
        )

    def get_data_for(self, label: QASectionLabel) -> Any:
        """
        Get the data of a label, in the shape its level stores it.

        Parameters
        ----------
        label : QASectionLabel
            Label of the data, e.g. ``ManifestSchema.METADATA_SUMMARY.MS``.

        Returns
        -------
        Any
            Data of the label, in the shape of its level and with values
            of its schema, as checked when written. None if none is
            recorded.
        """
        return label.load(self.sections.get(label.section, {}))

    def to_dict(self) -> dict:
        """
        Convert the field metadata to a JSON serialisable dict.

        Returns
        -------
        dict
            Field metadata, with the same keys as the constructor
            parameters.
        """
        return {
            "qa_dir": self.qa_dir,
            "field_name": self.field_name,
            "sections": self.sections,
        }
