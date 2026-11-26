"""
Registry of the QA report sections: their builders, by section type,
and the dispatch to them.

A builder is registered for a section type with
``@section_registry(SectionType)`` on a SectionRegistry, and run with
``section_registry.build(SectionType, field_metadata)``, which shows a run
metadata error as an error section.
The builders of the report and its one SectionRegistry,
``section_registry``, are in :mod:`.sections`.
"""

from typing import Callable, Concatenate

from ..exceptions import QAReportException, RunMetadataError
from ..repository.field_metadata import FieldMetadata
from ..view.sections import ErrorSection, Section

Builder = Callable[Concatenate[FieldMetadata, ...], Section]


class SectionRegistry:
    """
    Builders of the sections of the QA report, by section type.

    Builders are registered with :meth:`builds`, also run by calling the
    instance as a decorator, and run with :meth:`build`. The one instance
    used by the report, ``section_registry``, is in
    :mod:`.sections`, with the builders it registers:
    ``@section_registry(SectionType)`` registers the builder of a section
    and ``section_registry.build(SectionType, field_metadata)`` builds it.
    Each instance has its own registry, so tests can use a fresh one.
    """

    def __init__(self):
        self._builders: dict[type[Section], Builder] = {}

    def builds(
        self, section_type: type[Section]
    ) -> Callable[[Builder], Builder]:
        """
        Register the decorated function as the builder of a section type.

        A builder takes the run metadata of a field, returns its section
        and raises RunMetadataError if a product it needs is not recorded
        or cannot be read. The decorator only registers the builder: it is
        returned unchanged, and :meth:`build` turns its errors into an
        error section. Calling the instance does the same, so builders are
        decorated with ``@section_registry(SectionType)``.

        Parameters
        ----------
        section_type : type[Section]
            Type of the section the builder builds, e.g.
            ``FlaggingStatisticsSection``.

        Returns
        -------
        Callable
            Decorator registering a builder of ``section_type``.

        Raises
        ------
        TypeError
            If ``section_type`` is not a Section type, e.g. when used
            without a section type as ``@section_registry``.
        ValueError
            If a builder is already registered for ``section_type``.

        Examples
        --------
        Not a doctest: running it would register a second builder.

        .. code-block:: python

            @section_registry(FlaggingStatisticsSection)
            def flagging_statistics_section(field_metadata):
                ...
        """
        # Checked on import, so a misuse fails before any report is built
        if not (
            isinstance(section_type, type)
            and issubclass(section_type, Section)
        ):
            raise TypeError(
                f"A builder registers a Section type, not " f"{section_type!r}"
            )

        def register(builder: Builder) -> Builder:
            if section_type in self._builders:
                raise ValueError(
                    f"{section_type.__name__} already has a builder: "
                    f"{self._builders[section_type].__name__}"
                )
            self._builders[section_type] = builder
            return builder

        return register

    def build(
        self,
        section_type: type[Section],
        field_metadata: FieldMetadata,
        **kwargs,
    ) -> Section:
        """
        Build a section of the QA report of a field from its run metadata.

        Only run metadata errors, i.e. missing or unreadable products, are
        shown as an error section. Any other error is a bug, so it is
        raised rather than shown in the report.

        Parameters
        ----------
        section_type : type[Section]
            Type of the section to build, e.g. ``MetadataSection``.
        field_metadata : FieldMetadata
            Run metadata of the field.

        Returns
        -------
        Section
            Section of the type given, or an ErrorSection with its title if
            a product it needs is not recorded or cannot be read.

        Raises
        ------
        KeyError
            If no builder is registered for the section type.
        """
        builder = self._builders[section_type]
        try:
            return builder(field_metadata, **kwargs)
        except RunMetadataError as error:
            return ErrorSection(
                section_type.title,
                QAReportException(
                    f"{section_type.title} of field "
                    f"{field_metadata.field_name} cannot be built",
                    cause=error,
                ),
            )

    __call__ = builds
