from mock import Mock, patch

from ska_sdp_instrumental_calibration.scheduler import UpstreamOutput
from ska_sdp_instrumental_calibration.stages._utils import (
    _create_path_tree,
    get_gaintables_path,
    get_initialisation_vis,
    get_plots_path,
    get_visibilities_path,
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


def test_should_get_vis_for_initialisation_without_beam_normalisation():
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")

    assert get_initialisation_vis(upstream_output) == (
        upstream_output.vis,
        upstream_output.modelvis,
        False,
    )

    upstream_output["beam_normalised_vis"] = None
    upstream_output["beam_normalised_modelvis"] = None

    assert get_initialisation_vis(upstream_output) == (
        upstream_output.vis,
        upstream_output.modelvis,
        False,
    )


def test_should_get_beam_normalised_vis_for_initialisation():
    upstream_output = UpstreamOutput()
    upstream_output["vis"] = Mock(name="vis")
    upstream_output["modelvis"] = Mock(name="modelvis")
    upstream_output["beam_normalised_vis"] = Mock(name="normalised vis")
    upstream_output["beam_normalised_modelvis"] = Mock(
        name="normalised modelvis"
    )

    assert get_initialisation_vis(upstream_output) == (
        upstream_output.beam_normalised_vis,
        upstream_output.beam_normalised_modelvis,
        True,
    )
