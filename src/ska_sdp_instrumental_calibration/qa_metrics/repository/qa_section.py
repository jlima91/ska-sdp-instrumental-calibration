"""
QA sections of the run metadata and their data labels, e.g.
``QASection.METADATA_SUMMARY.MS``.

Each label declares a :class:`LabelSpec`: its :class:`Level` in the
field hierarchy, which decides how its data is stored, and the schema of
one entry.
"""

import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, TypeAlias

from ..exceptions import RunMetadataError

# Any value json.dump can write: what the run metadata stores. Static
# documentation only; entries are checked at runtime against the schema of
# their label. Mapping and Sequence (covariant, unlike dict and list) let
# e.g. dict[str, str] or list[int] count as JSONSerialisable. A TypedDict
# does not: type checkers only accept it as Mapping[str, object].
# Only the self-references are quoted, so the alias resolves in any module
# that imports it, e.g. with typing.get_type_hints
JSONSerialisable: TypeAlias = (
    Mapping[str, "JSONSerialisable"]
    | Sequence["JSONSerialisable"]
    | str
    | int
    | float
    | bool
    | None
)


class Level(Enum):
    """
    Level of a label in the run metadata hierarchy of a field.

    ``FIELD_SINGLE``
        One value for the field; a write overwrites it. Stored as
        ``value``.
    ``FIELD_MULTIPLE``
        Several values for the field; a write appends. Stored as
        ``[value, ...]``.
    ``PREFIX_SINGLE``
        One value per MS prefix, for data written once per MS, e.g. by
        load_data or a calibration stage; a write overwrites the value of
        its prefix. Stored as ``{prefix: value}``. Stages that run more
        than once per MS, as in the experimental pipeline, are not
        supported: only their last write is kept.
    """

    FIELD_SINGLE = "field_single"
    FIELD_MULTIPLE = "field_multiple"
    PREFIX_SINGLE = "prefix_single"


@dataclass(frozen=True)
class LabelSpec:
    """
    Storage level and entry schema of a label.

    Parameters
    ----------
    level : Level
        Level of the label in the field hierarchy.
    schema : Any, optional
        Type of one entry, e.g. ``str``, ``int``, ``list[float]`` or a
        ``TypedDict``. Kept JSON native so the run metadata reads back
        with the same shape; paths are stored as ``str``. Defaults to
        ``str``.
    """

    level: Level
    schema: Any = str

    @property
    def runtime_type(self) -> type:
        """
        Type an entry is checked against: the outer type of the schema.

        Returns
        -------
        type
            ``dict`` for a ``TypedDict``, the origin of a generic alias
            (e.g. ``list`` for ``list[float]``), else the schema itself.
        """
        # A TypedDict cannot be used with isinstance; its values are dicts
        if typing.is_typeddict(self.schema):
            return dict

        # Generic aliases such as list[float] are checked by their origin
        return typing.get_origin(self.schema) or self.schema

    def validate(
        self,
        label: str,
        data: JSONSerialisable,
        prefix: str | None = None,
    ):
        """
        Check an entry to write, and its prefix, against the spec.

        Parameters
        ----------
        label : str
            Name of the label, used in the error message, e.g.
            ``flagging.gaintable``.
        data : JSONSerialisable
            Entry to write, checked against the schema.
        prefix : str, optional
            MS prefix of the entry. Required for the prefix levels, not
            allowed for the field levels.

        Raises
        ------
        RunMetadataError
            If the prefix is missing or not allowed, or the entry is not
            of the runtime type of the schema.
        """
        needs_prefix = self.level == Level.PREFIX_SINGLE

        if needs_prefix and prefix is None:
            raise RunMetadataError(
                f"{label} is stored per MS and needs a prefix"
            )
        if not needs_prefix and prefix is not None:
            raise RunMetadataError(
                f"{label} is stored per field and takes no prefix"
            )

        expected = self.runtime_type
        if not isinstance(data, expected):
            raise RunMetadataError(
                f"{label} expects {expected.__name__}, "
                f"got {type(data).__name__}"
            )


class QASectionLabel(StrEnum):
    """
    Base for the data labels of a QA section in the run metadata.

    Members are declared as ``NAME = "value", LabelSpec(...)``. The value
    stays a plain string and the spec is kept on the member as ``spec``.

    The section key is derived from the subclass name, e.g.
    ``METADATA_SUMMARY`` -> ``metadata_summary``.
    """

    spec: LabelSpec

    def __new__(cls, value: str, spec: LabelSpec):
        # StrEnum only accepts a str value, so the (value, spec) tuple of
        # a member is split here and the spec kept as an attribute
        member = str.__new__(cls, value)
        member._value_ = value
        member.spec = spec
        return member

    @property
    def section(self) -> str:
        """
        Key of the QA section in the run metadata.

        Returns
        -------
        str
            Lower-cased name of the QA section class.
        """
        return type(self).__name__.lower()

    @property
    def qualified_name(self) -> str:
        """
        Name of the label within its section, for messages.

        Returns
        -------
        str
            ``<section>.<label>``, e.g. ``flagging.gaintable``.
        """
        return f"{self.section}.{self}"

    def store(
        self,
        section: dict,
        data: JSONSerialisable,
        prefix: str | None = None,
    ):
        """
        Store an entry in the section, as the level of the label says.

        Parameters
        ----------
        section : dict
            Data of the QA section of the label, keyed by label.
        data : JSONSerialisable
            Entry to store, of the schema of the label.
        prefix : str, optional
            MS prefix of the entry. Required for prefix level labels, not
            allowed for field level labels.

        Raises
        ------
        RunMetadataError
            If the prefix is missing or not allowed, or the entry does
            not match the schema of the label.
        """
        self.spec.validate(self.qualified_name, data=data, prefix=prefix)

        match self.spec.level:
            case Level.FIELD_SINGLE:
                section[self] = data
            case Level.FIELD_MULTIPLE:
                section.setdefault(self, []).append(data)
            case Level.PREFIX_SINGLE:
                section.setdefault(self, {})[prefix] = data

    def load(self, section: dict) -> Any:
        """
        Get the stored data of the label from the section, in the shape
        its level stores it. Consumers reshape it as they need.

        Parameters
        ----------
        section : dict
            Data of the QA section of the label, keyed by label.

        Returns
        -------
        Any
            ``value`` for ``FIELD_SINGLE``, ``[value, ...]`` for
            ``FIELD_MULTIPLE`` and ``{prefix: value}`` for
            ``PREFIX_SINGLE``, with values of the schema of the label.
            None if nothing is stored.
        """
        # Any, not JSONSerialisable: the schema of the label was checked
        # when the data was written, so consumers can rely on it
        return section.get(self)


class QASection:
    """
    Namespace of the QA sections in the run metadata, e.g.
    ``QASection.METADATA_SUMMARY.MS``.
    """

    class METADATA_SUMMARY(QASectionLabel):
        """Data labels of the ``metadata_summary`` QA section."""

        MS = "MS", LabelSpec(Level.FIELD_MULTIPLE, str)
        TIMESTAMPS = "timestamps", LabelSpec(Level.FIELD_MULTIPLE, list[float])
        # Only one writer per field: data_exports groups by field
        FINAL_GAINTABLE = "final_gaintable", LabelSpec(Level.FIELD_SINGLE, str)
        # Can be overwritten, as its source of truth does not change
        CHANNELS = "channels", LabelSpec(Level.FIELD_SINGLE, int)
        STATION_CONFIGURATION = "stations", LabelSpec(
            Level.FIELD_SINGLE, dict[str, str]
        )

    class FLAGGING(QASectionLabel):
        """Data labels of the ``flagging`` QA section."""

        # Only load_data writes it, once per MS: it cannot be repeated
        PRE_FLAGGING = "pre_flagging", LabelSpec(Level.PREFIX_SINGLE, str)
        GAIN_FITS = "gain_fits", LabelSpec(Level.PREFIX_SINGLE, str)
        GAINTABLE = "gaintable", LabelSpec(Level.PREFIX_SINGLE, str)

    class BANDPASS(QASectionLabel):
        """Data labels of the ``bandpass`` QA section."""

        GAINTABLE = "gaintable", LabelSpec(Level.PREFIX_SINGLE, str)

    class DELAY(QASectionLabel):
        """Data labels of the ``delay`` QA section."""

        GAINTABLE = "gaintable", LabelSpec(Level.PREFIX_SINGLE, str)
        DELAYTABLE = "delaytable", LabelSpec(Level.PREFIX_SINGLE, str)
