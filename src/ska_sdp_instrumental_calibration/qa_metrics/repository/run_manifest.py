"""
Run metadata of an INST run: the data products recorded per field, used to
generate the QA report.

The metadata is grouped per field, then per QA section, then per label.
How a label stores its data, and what data it accepts, is set by its
``LabelSpec`` (see :mod:`.manifest_schema`), e.g.::

    {
        "<field>": {
            "qa_dir": "<qa_dir>",
            "field_name": "<field>",
            "sections": {
                "metadata_summary": {
                    "MS": ["a.ms", "b.ms"],
                    "channels": 64
                },
                "bandpass": {
                    "gaintable": {"a": ["a/bandpass.gaintable.h5parm"]}
                }
            }
        }
    }
"""

import json
import os

from ..exceptions import RunMetadataError
from .field_metadata import FieldMetadata
from .manifest_schema import JSONSerialisable, QASectionLabel


class RunManifest:
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

    def __iter__(self):
        return iter(self.field_metadata.values())

    def add_for_field(
        self,
        field_name,
        label: QASectionLabel,
        data: JSONSerialisable,
        prefix: str | None = None,
    ):
        """
        Add data to a label of a field. The label validates the data and
        decides whether it overwrites or appends.

        Parameters
        ----------
        field_name : str
            Field name.
        label : QASectionLabel
            Label of the data, e.g. ``ManifestSchema.METADATA_SUMMARY.MS``.
        data : JSONSerialisable
            Data of the schema of the label.
        prefix : str, optional
            MS prefix of the data, for labels stored per MS.
        """
        field_metadata = self.field_metadata.setdefault(
            field_name, FieldMetadata(field_name, self.qa_dir)
        )
        field_metadata.add_data_for(label, data, prefix=prefix)

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
    def initialize(cls, qa_dir) -> "RunManifest":
        """
        Read the run metadata from ``<qa_dir>/run_metadata.json``.

        Parameters
        ----------
        qa_dir : str
            QA directory of the run.

        Returns
        -------
        RunManifest
            The run metadata.
        """
        path = os.path.join(qa_dir, cls.RUN_METADATA_PATH)

        if not os.path.exists(path):
            raise RunMetadataError(f"No run metadata found in {qa_dir}")

        with open(path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

            run_metadata = cls(qa_dir)
            run_metadata.field_metadata = {
                name: FieldMetadata(**data)
                for name, data in metadata_dict.items()
            }

            return run_metadata
