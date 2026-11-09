from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages._utils import (
    _create_path_tree,
    get_gaintables_path,
    get_plots_path,
    get_value_or_fallback,
    get_visibilities_path,
    has_value,
)


@patch("ska_sdp_instrumental_calibration.stages._utils.Path")
def test_should_create_path_tree(path_mock):
    path_mock.return_value = path_mock
    _create_path_tree("/output/path")

    path_mock.assert_called_once_with("/output/path")
    path_mock.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch("ska_sdp_instrumental_calibration.stages._utils._create_path_tree")
def test_should_get_gaintables_path(create_path_tree_mock):
    result = get_gaintables_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with("/output/gaintables/prefix")

    assert result == "/output/gaintables/prefix"


@patch("ska_sdp_instrumental_calibration.stages._utils._create_path_tree")
def test_should_get_plots_path(create_path_tree_mock):
    result = get_plots_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with("/output/plots/prefix")

    assert result == "/output/plots/prefix"


@patch("ska_sdp_instrumental_calibration.stages._utils._create_path_tree")
def test_should_get_visibilities_path(create_path_tree_mock):
    result = get_visibilities_path("/output", "prefix")
    create_path_tree_mock.assert_called_once_with(
        "/output/visibilities/prefix"
    )

    assert result == "/output/visibilities/prefix"


def test_should_check_if_upstream_output_has_value():
    upstream_output = UpstreamOutput()
    upstream_output["none_key"] = None
    upstream_output["value_key"] = Mock(name="value")

    assert has_value(upstream_output, "absent_key") is False
    assert has_value(upstream_output, "none_key") is False
    assert has_value(upstream_output, "value_key") is True


def test_should_get_value_when_key_has_value():
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["beam_normalised_vis"] = Mock(name="normalised vis")

    assert (
        get_value_or_fallback(upstream_output, "beam_normalised_vis", "vis")
        == upstream_output["beam_normalised_vis"]
    )


def test_should_get_fallback_value_when_key_is_absent_or_none():
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")

    assert (
        get_value_or_fallback(upstream_output, "beam_normalised_vis", "vis")
        == upstream_output.vis
    )

    upstream_output["beam_normalised_vis"] = None

    assert (
        get_value_or_fallback(upstream_output, "beam_normalised_vis", "vis")
        == upstream_output.vis
    )
