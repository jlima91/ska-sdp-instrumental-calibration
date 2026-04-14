"""Test for sdm"""

import tempfile
from pathlib import Path

import pytest
from mock import call, patch

from ska_sdp_instrumental_calibration.sdm import SDM


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_initialise_sdm_for_the_given_sdm_root(path_mock):
    """test_should_initialise_sdm_for_the_given_sdm_root"""
    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    sdm_root = "/path/to/sdm/root"
    SDM.init(sdm_root)

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("sky"),
            call("telmodel"),
            call("telmodel/instrument"),
            call("calibration/gains"),
            call("calibration/pointing"),
            call("calibration/bandpass"),
            call("logs"),
        ]
    )

    path_mock.mkdir.assert_any_call(parents=True, exist_ok=True)
    assert path_mock.mkdir.call_count == 7


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_return_paths_of_model_contents_for_a_given_field_id(path_mock):
    """test_should_return_paths_of_model_contents_for_a_given_field_id"""
    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    sdm_root = "/path/to/sdm/root"
    path_mock.iterdir.return_value = [Path("file_1.txt"), Path("file_2.txt")]

    sdm_paths = SDM.SKY.find_models(sdm_root, "target")

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("sky"),
            call("target"),
        ]
    )

    path_mock.iterdir.assert_called_once()
    assert sdm_paths == [Path("file_1.txt"), Path("file_2.txt")]


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_fileter_paths_of_model_contents_for_a_given_field_id(
    path_mock,
):
    """test_should_fileter_paths_of_model_contents_for_a_given_field_id"""
    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    sdm_root = "/path/to/sdm/root"
    path_mock.rglob.return_value = [
        Path("file_1.csv"),
    ]

    sdm_paths = SDM.SKY.find_models(sdm_root, "target", "*.csv")

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("sky"),
            call("target"),
        ]
    )

    path_mock.rglob.assert_called_once_with("*.csv")
    assert sdm_paths == [Path("file_1.csv")]


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_return_sky_model_for_all_fields(path_mock):
    """test_should_return_sky_model_for_all_fields"""
    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    sdm_root = "/path/to/sdm/root"
    path_mock.rglob.return_value = [
        Path("file_1.csv"),
        Path("file_2.csv"),
    ]

    sdm_paths = SDM.SKY.find_models(sdm_root)

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("sky"),
        ]
    )

    path_mock.rglob.assert_called_once_with("*")
    assert sdm_paths == [
        Path("file_1.csv"),
        Path("file_2.csv"),
    ]


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_return_specific_model_for_a_given_field(path_mock):
    """test_should_return_specific_model_for_a_given_field"""
    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    sdm_root = "/path/to/sdm/root"
    path_mock.rglob.return_value = [
        Path("file_1.csv"),
        Path("file_2.csv"),
    ]

    SDM.SKY.find_model(sdm_root, "target", "file*.csv")

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("sky"),
            call("target"),
        ]
    )

    path_mock.rglob.assert_called_once_with("file*.csv")

    path_mock.rglob.return_value = []
    assert SDM.SKY.find_model(sdm_root, "target", "file*.csv") is None


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_create_field_and_give_new_model_path(path_mock):
    """test should create field and give new model path"""

    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    path_mock.exists.return_value = False

    sdm_root = "/path/to/sdm/root"
    SDM.GAINS.prepare_model(sdm_root, "calibrator", "gaintable.h5parm")

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("calibration/gains"),
            call("calibrator"),
            call("gaintable.h5parm"),
        ]
    )
    path_mock.mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_create_log_path(path_mock):
    """test should create field and give new model path"""

    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    path_mock.iterdir.return_value = [
        Path("slurm.out"),
        Path("01-bpp"),
        Path("02-inst"),
    ]

    sdm_root = "/path/to/sdm/root"
    SDM.prepare_log_dir(sdm_root, "inst")

    path_mock.assert_called_once_with(sdm_root)
    path_mock.__truediv__.assert_has_calls(
        [
            call("logs"),
            call("03-inst"),
        ]
    )
    path_mock.mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_should_return_most_recent_log_path():
    """test should get log folders"""

    with tempfile.TemporaryDirectory() as temp_dir:
        sdm_root = Path(f"{temp_dir}/sdm")

        SDM.init(sdm_root)
        SDM.prepare_log_dir(sdm_root, "bpp")
        assert SDM.get_log_dir(sdm_root, "inst") == Path(
            f"{temp_dir}/sdm/logs/02-inst"
        )
        SDM.prepare_log_dir(sdm_root, "bpp")
        SDM.prepare_log_dir(sdm_root, "inst")

        assert SDM.get_log_dir(sdm_root, "inst") == Path(
            f"{temp_dir}/sdm/logs/04-inst"
        )


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_prepare_new_sdm_path(path_mock):
    """test should create field and give new model path"""

    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    path_mock.exists.return_value = False

    sdm_root = "/path/to/sdm/root"
    SDM.GAINS.prepare_model(sdm_root, "calibrator", "gaintable.h5parm")

    path_mock.assert_called_once_with(sdm_root)
    assert path_mock.__truediv__.mock_calls == [
        call("calibration/gains"),
        call("calibrator"),
        call("gaintable.h5parm"),
    ]


@patch("ska_sdp_instrumental_calibration.sdm.shutil")
@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_backup_existing_model_and_provide_path(path_mock, shutil_mock):
    """test should create field and give new model path"""

    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    path_mock.exists.return_value = True
    path_mock.glob.return_value = [
        Path("gaintable.h5parm"),
        Path("01-gaintable.h5parm"),
        Path("02-gaintable.h5parm"),
    ]

    sdm_root = "/path/to/sdm/root"
    SDM.GAINS.prepare_model(sdm_root, "calibrator", "gaintable.h5parm")

    path_mock.assert_called_once_with(sdm_root)

    assert path_mock.__truediv__.mock_calls == [
        call("calibration/gains"),
        call("calibrator"),
        call("gaintable.h5parm"),
        call("03-gaintable.h5parm"),
    ]

    path_mock.glob.assert_called_once_with("*gaintable.h5parm")

    shutil_mock.move.assert_called_once_with(path_mock, path_mock)


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_raise_exception_for_put__log(path_mock):
    """test should create field and give new model path"""

    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    path_mock.exists.return_value = True

    sdm_root = "/path/to/sdm/root"
    with pytest.raises(
        RuntimeError, match="Use SDM.prepare_log_dir to prepare log directory"
    ):
        SDM.LOGS.prepare_model(sdm_root, "inst", None)


@patch("ska_sdp_instrumental_calibration.sdm.Path")
def test_should_raise_exception_for_put_other_than_log(path_mock):
    """test should create field and give new model path"""

    path_mock.return_value = path_mock
    path_mock.__truediv__.return_value = path_mock
    path_mock.exists.return_value = True

    sdm_root = "/path/to/sdm/root"
    with pytest.raises(TypeError, match="Model name not provided"):
        SDM.GAINS.prepare_model(sdm_root, "calibrator", None)


@patch("ska_sdp_instrumental_calibration.sdm.shutil")
def test_should_clone_sdm_from_an_existing_sdm(shutil_mock):
    """test_should_clone_sdm_from_an_existing_sdm"""
    existing_sdm_root = "/path/to/existing/sdm/root"
    new_sdm_root = "/path/to/new/sdm/root"
    SDM.clone(existing_sdm_root, new_sdm_root)

    shutil_mock.copytree.assert_called_once_with(
        existing_sdm_root, new_sdm_root
    )


def test_should_manage_sdm_lifecycle():
    """integration test for SDM management"""

    with tempfile.TemporaryDirectory() as temp_dir:
        sdm_root = Path(f"{temp_dir}/sdm")

        SDM.init(sdm_root)
        SDM.prepare_log_dir(sdm_root, "bpp")
        SDM.prepare_log_dir(sdm_root, "inst")

        log_paths = SDM.LOGS.find_models(sdm_root)

        assert len(log_paths) == 2

        assert Path(f"{sdm_root}/logs/01-bpp") in log_paths
        assert Path(f"{sdm_root}/logs/02-inst") in log_paths

        for sdm in [SDM.GAINS, SDM.BANDPASS, SDM.POINTING, SDM.SKY]:

            model_path = sdm.prepare_model(
                sdm_root, "field-id", "sdm-file.ext"
            )
            assert not model_path.exists()
            model_path.touch()

            existing_model_path = sdm.find_model(
                sdm_root, "field-id", "sdm-file.ext"
            )
            assert existing_model_path == model_path

            new_model_path = sdm.prepare_model(
                sdm_root, "field-id", "sdm-file.ext"
            )
            assert not new_model_path.exists()

            new_model_path.touch()

            models = sdm.find_models(sdm_root, "field-id")

            assert len(models) == 2

            assert (
                Path(f"{sdm_root}/{sdm.value}/field-id/sdm-file.ext") in models
            )
            assert (
                Path(f"{sdm_root}/{sdm.value}/field-id/01-sdm-file.ext")
                in models
            )

        new_sdm_root = f"{temp_dir}/sdm-new"
        SDM.clone(sdm_root, new_sdm_root)

        original_sdms = {
            sdm.relative_to(sdm_root) for sdm in sdm_root.rglob("*")
        }
        cloned_sdms = {
            sdm.relative_to(new_sdm_root)
            for sdm in Path(new_sdm_root).rglob("*")
        }

        assert original_sdms == cloned_sdms
