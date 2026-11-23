import pytest
from mock import Mock

from ska_sdp_instrumental_calibration.qa_metrics.exceptions import (
    RunMetadataError,
)
from ska_sdp_instrumental_calibration.qa_metrics.field_sections import (
    section_builder,
)
from ska_sdp_instrumental_calibration.qa_metrics.run_metadata import (
    FieldMetadata,
)
from ska_sdp_instrumental_calibration.qa_metrics.section_builder import (
    SectionBuilder,
)
from ska_sdp_instrumental_calibration.qa_metrics.sections import (
    CableDelaysSection,
    ErrorSection,
    FlaggingStatisticsSection,
    MetadataSection,
)


@pytest.fixture
def builder():
    """
    A fresh section builder: tests register on it, not on the report's
    section_builder
    """
    return SectionBuilder()


@pytest.fixture
def field_metadata():
    return FieldMetadata("field_a", "/qa/dir", sections={})


def test_should_register_the_builders_of_the_report():
    # pylint: disable-next=protected-access
    assert set(section_builder._builders) == {
        MetadataSection,
        FlaggingStatisticsSection,
        CableDelaysSection,
    }


def test_should_start_with_no_builders():
    # Each instance has its own registry: the report's is not shared
    # pylint: disable-next=protected-access
    assert SectionBuilder()._builders == {}


def test_should_build_section_with_its_registered_builder(
    builder, field_metadata
):
    section = Mock(name="section")
    metadata_builder = Mock(return_value=section)
    builder(MetadataSection)(metadata_builder)

    assert builder.build(MetadataSection, field_metadata) is section
    metadata_builder.assert_called_once_with(field_metadata)


def test_should_return_the_decorated_builder_unchanged(builder):
    def _metadata_section(field_metadata):
        return field_metadata

    assert builder(MetadataSection)(_metadata_section) is _metadata_section


def test_should_register_with_builds_as_with_calling_the_instance(
    builder, field_metadata
):
    section = Mock(name="section")
    builder.builds(MetadataSection)(Mock(return_value=section))

    assert builder.build(MetadataSection, field_metadata) is section


def test_should_get_error_section_for_run_metadata_errors(
    builder, field_metadata
):
    error = RunMetadataError("Gaintable a.h5parm was not produced")
    builder(MetadataSection)(Mock(side_effect=error))

    section = builder.build(MetadataSection, field_metadata)

    assert isinstance(section, ErrorSection)
    assert section.title == MetadataSection.title
    assert str(section.error) == (
        "Metadata summary of field field_a cannot be built"
    )
    assert section.error.__cause__ is error


def test_should_raise_errors_other_than_run_metadata_errors(
    builder, field_metadata
):
    # Any other error is a bug: it must fail loudly, not hide in the report
    builder(MetadataSection)(
        Mock(side_effect=ValueError("bug in the computation"))
    )

    with pytest.raises(ValueError, match="bug in the computation"):
        builder.build(MetadataSection, field_metadata)


def test_should_raise_if_no_builder_for_section_type(builder, field_metadata):
    with pytest.raises(KeyError):
        builder.build(MetadataSection, field_metadata)


def test_should_raise_if_used_without_section_type(builder):
    # @builder without a section type passes the builder as section type
    def _metadata_section(field_metadata):
        return field_metadata

    with pytest.raises(TypeError, match="registers a Section type"):
        builder(_metadata_section)


def test_should_raise_if_section_type_already_has_a_builder(builder):
    def _metadata_section(field_metadata):
        return field_metadata

    builder(MetadataSection)(_metadata_section)

    with pytest.raises(
        ValueError, match="MetadataSection already has a builder"
    ):
        builder(MetadataSection)(Mock(__name__="_other_section"))
