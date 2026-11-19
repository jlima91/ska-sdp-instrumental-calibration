"""
Run metadata of an INST run: the data products recorded per field, used to
generate the QA report.

The metadata is grouped per field, then per QA section, e.g.::

    {
        "<field>": {
            "qa_dir": "<qa_dir>",
            "field_name": "<field>",
            "sections": {"metadata_summary": {"MS": ["a.ms", "b.ms"]}}
        }
    }
"""

import json
import os
from typing import Any

from ..exceptions import QAMetadataError
from .qa_section import QASectionLabel

# Value type accepted for run metadata entries
DataProduct = str | os.PathLike[str]


class FieldMetadata:
    """
    Data products recorded for one field.

    Each field gets its own QA report, and a ``FieldMetadata`` holds
    everything needed to generate it, including the QA directory.

    Paths are stored as ``str`` (``os.fspath``) so that the metadata stays
    JSON serialisable and reads back with the same type.

    Parameters
    ----------
    field_name : str
        Field name.
    qa_dir : str
        QA directory of the run.
    sections : dict[str, dict[str, list[DataProduct]]], optional
        Data products keyed by QA section, then by label. Empty if not
        given.
    """

    def __init__(
        self,
        field_name,
        qa_dir,
        sections: dict[str, dict[str, list[DataProduct]]] | None = None,
    ):
        self.field_name = field_name
        # str so the field metadata stays JSON serialisable
        self.qa_dir = os.fspath(qa_dir)
        self.sections = {} if sections is None else sections

    def add_data_for(self, label: QASectionLabel, data: list[DataProduct]):
        """
        Set the data products of a label, replacing any recorded ones.

        Parameters
        ----------
        label : QASectionLabel
            Label of the data products, e.g. ``QASection.METADATA_SUMMARY.MS``.
        data : list[DataProduct]
            Data products of the label.
        """
        self.sections.setdefault(label.section, {})[label] = [
            path for path in data
        ]

    def append_data_for(self, label: QASectionLabel, data: DataProduct):
        """
        Add a data product to a label, keeping the recorded ones.

        Parameters
        ----------
        label : QASectionLabel
            Label of the data product, e.g. ``QASection.METADATA_SUMMARY.MS``.
        data : DataProduct
            Data product to add.
        """
        section = self.sections.setdefault(label.section, {})
        section.setdefault(label, []).append(data)

    def get_data_for(self, label: QASectionLabel) -> list[Any] | None:
        """
        Get the data products of a label.

        Parameters
        ----------
        label : QASectionLabel
            Label of the data products, e.g. ``QASection.METADATA_SUMMARY.MS``.

        Returns
        -------
        list[DataProduct] or None
            Data products of the label, or None if none are recorded.
        """
        return self.sections.get(label.section, {}).get(label)

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


class RunMetadata:
    """
    Data products of an INST run, per field, written to
    ``<qa_dir>/run_metadata.json``.

    Parameters
    ----------
    qa_dir : str
        QA directory of the run, where the run metadata is written.
    """

    RUN_METADATA_PATH = "run_metadata.json"

    def __init__(self, qa_dir):
        self.qa_dir = qa_dir
        self.field_metadata: dict[str, FieldMetadata] = {}

    def add_for_field(
        self, field_name, label: QASectionLabel, data: list[DataProduct]
    ):
        """
        Set the data products of a label for a field, replacing any
        recorded ones.

        Parameters
        ----------
        field_name : str
            Field name.
        label : QASectionLabel
            Label of the data products, e.g. ``QASection.METADATA_SUMMARY.MS``.
        data : list[DataProduct]
            Data products of the label.
        """
        field_metadata = self.field_metadata.setdefault(
            field_name, FieldMetadata(field_name, self.qa_dir)
        )
        field_metadata.add_data_for(label, data)

    def __iter__(self):
        return iter(self.field_metadata.values())

    def append_for_field(self, field_name, label: QASectionLabel, data: Any):
        """
        Add a data product to a label for a field, keeping the recorded
        ones.

        Parameters
        ----------
        field_name : str
            Field name.
        label : QASectionLabel
            Label of the data product, e.g. ``QASection.METADATA_SUMMARY.MS``.
        data : DataProduct
            Data product to add.
        """
        field_metadata = self.field_metadata.setdefault(
            field_name, FieldMetadata(field_name, self.qa_dir)
        )
        field_metadata.append_data_for(label, data)

    def to_json(self):
        """
        Write the run metadata to ``<qa_dir>/run_metadata.json``.
        """
        metadata_dict = {
            name: metadata.to_dict()
            for name, metadata in self.field_metadata.items()
        }

        path = os.path.join(self.qa_dir, self.RUN_METADATA_PATH)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f)

    @classmethod
    def initialize(cls, qa_dir) -> "RunMetadata":
        """
        Read the run metadata from ``<qa_dir>/run_metadata.json``.

        Parameters
        ----------
        qa_dir : str
            QA directory of the run.

        Returns
        -------
        RunMetadata
            The run metadata.
        """
        path = os.path.join(qa_dir, cls.RUN_METADATA_PATH)

        if not os.path.exists(path):
            raise QAMetadataError(f"No run metadata found in {qa_dir}")

        with open(path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

            run_metadata = cls(qa_dir)
            run_metadata.field_metadata = {
                name: FieldMetadata(**data)
                for name, data in metadata_dict.items()
            }

            return run_metadata
