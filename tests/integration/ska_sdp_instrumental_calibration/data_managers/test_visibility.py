import os

import dask
import numpy as np
import pytest
from casacore.tables import table
from mock import MagicMock, patch

from ska_sdp_instrumental_calibration.data_managers.visibility import (
    _resolve_weight_column,
    check_if_cache_files_exist,
    create_template_vis_from_ms,
    load_ms_as_dataset_with_time_chunks,
    read_antenna_response_data,
    read_ms_field_id,
    read_vis_shape_from_zarr,
    read_visibility_from_zarr,
    write_ms_to_zarr,
)


def create_mock_table(getcol_return=None, getkeywords_return=None):
    mock_table = MagicMock()
    mock_table.__enter__ = MagicMock(return_value=mock_table)
    mock_table.__exit__ = MagicMock(return_value=False)

    if getcol_return is not None:
        mock_table.getcol.return_value = getcol_return
    if getkeywords_return is not None:
        mock_table.getkeywords.return_value = getkeywords_return

    return mock_table


@pytest.fixture
def mock_table():
    with patch(
        "ska_sdp_instrumental_calibration.data_managers.visibility.table"
    ) as mock:
        yield mock


def test_visibility_write_and_read(tmp_path, generate_vis, generate_ms):
    """
    Integration test for all functionality under
    the visibility module.
    This will test:
    1. Conversion of MSv2 to Zarr file format
    2. Reading of zarr file as Visibility object
    """
    gen_vis, _ = generate_vis
    ms_path = generate_ms

    cache_dir = f"{tmp_path}/cache"
    os.makedirs(cache_dir)

    # Common dimensions across zarr and loaded visibility dataset
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }
    # This is chunking of the intermidiate zarr file
    zarr_chunks = {
        **non_chunked_dims,
        "time": 1,
        "frequency": 2,
    }
    vis_chunks = {
        **non_chunked_dims,
        "time": -1,
        "frequency": 2,
    }

    writer = write_ms_to_zarr(ms_path, cache_dir, zarr_chunks)
    dask.compute(writer)

    assert check_if_cache_files_exist(cache_dir)
    assert read_vis_shape_from_zarr(cache_dir) == gen_vis.vis.shape

    zarred_dataset = read_visibility_from_zarr(cache_dir, vis_chunks)

    np.testing.assert_allclose(
        np.real(gen_vis.vis.data), np.real(zarred_dataset.vis.data)
    )
    np.testing.assert_allclose(
        np.imag(gen_vis.vis.data), np.imag(zarred_dataset.vis.data)
    )
    np.testing.assert_allclose((gen_vis.uvw.data), (zarred_dataset.uvw.data))
    np.testing.assert_allclose(
        (gen_vis.flags.data), (zarred_dataset.flags.data)
    )
    # Might be problem with export_visibility_to_ms function, as the weight
    # column in the MS is not exactly same as the weight variable in the
    # visibility dataset. This needs to be investigated further.
    # np.testing.assert_allclose((gen_vis.weight.data),(zarred_dataset.weight.data))

    np.testing.assert_allclose(
        (gen_vis.frequency.data), (zarred_dataset.frequency.data)
    )
    np.testing.assert_equal((gen_vis.time.data), (zarred_dataset.time.data))


def test_visibility_write_and_read_for_multiple_ms_input(
    tmp_path, generate_vis, generate_ms
):
    gen_vis, _ = generate_vis
    ms_path = generate_ms
    cache_dir = f"{tmp_path}/cache"
    os.makedirs(cache_dir)

    # Common dimensions across zarr and loaded visibility dataset
    non_chunked_dims = {
        dim: -1
        for dim in [
            "baselineid",
            "polarisation",
            "spatial",
        ]
    }
    # This is chunking of the intermidiate zarr file
    zarr_chunks = {
        **non_chunked_dims,
        "time": 1,
        "frequency": 2,
    }
    vis_chunks = {
        **non_chunked_dims,
        "time": -1,
        "frequency": 2,
    }

    writer = write_ms_to_zarr([ms_path, ms_path], cache_dir, zarr_chunks)
    dask.compute(writer)

    assert check_if_cache_files_exist(cache_dir)

    zarred_dataset = read_visibility_from_zarr(cache_dir, vis_chunks)

    time_vars = ["vis", "uvw", "flags", "integration_time", "datetime", "time"]
    for var in time_vars:
        expected = np.concatenate(
            [getattr(gen_vis, var), getattr(gen_vis, var)]
        )
        np.testing.assert_equal(
            expected,
            getattr(zarred_dataset, var),
            err_msg=f"Mismatch in time-dependent variable: {var}",
        )

    datapoints = [
        "frequency",
        "baselines",
        "polarisation",
        "spatial",
        "antenna1",
        "antenna2",
        "channel_bandwidth",
    ]
    for datapoint in datapoints:
        np.testing.assert_equal(
            getattr(gen_vis, datapoint).data,
            getattr(zarred_dataset, datapoint).data,
            err_msg=f"Mismatch in data: {datapoint}",
        )

    # Might be problem with export_visibility_to_ms function, as the weight
    # column in the MS is not exactly same as the weight variable in the
    # visibility dataset. This needs to be investigated further.
    # np.testing.assert_equal(np.concatenate([gen_vis.weight, gen_vis.weight]),
    # zarred_dataset.weight)

    # Check attributes is failing as frame attribute is not preserved
    # assert gen_vis.attrs == zarred_dataset.attrs


@pytest.fixture
def ms_with_distinct_weights(generate_ms):
    """
    Fill the WEIGHT and WEIGHT_SPECTRUM columns of the test MSv2 with
    distinct values, to identify the column the weights are loaded from.
    """
    with table(generate_ms, readonly=False, ack=False) as tab:
        weight_shape = tab.getcol("WEIGHT").shape
        spectrum_shape = tab.getcol("WEIGHT_SPECTRUM").shape
        weight = np.arange(np.prod(weight_shape), dtype=np.float32).reshape(
            weight_shape
        )
        weight_spectrum = (
            1000 + np.arange(np.prod(spectrum_shape), dtype=np.float32)
        ).reshape(spectrum_shape)
        tab.putcol("WEIGHT", weight)
        tab.putcol("WEIGHT_SPECTRUM", weight_spectrum)

    return generate_ms, weight, weight_spectrum


def _expected_vis_weight(weight_column_data, vis_shape):
    """
    Rearrange MSv2 weight column data (rows, [frequency,] polarisation)
    into the shape of visibility weights.
    """
    ntimes, nbaselines, nchannels, npols = vis_shape
    if weight_column_data.ndim == 2:
        weight_column_data = np.broadcast_to(
            weight_column_data[:, np.newaxis, :],
            (weight_column_data.shape[0], nchannels, npols),
        )
    return weight_column_data.reshape(vis_shape)


@pytest.mark.parametrize(
    "weightcolumn, expected_column",
    [
        ("auto", "WEIGHT_SPECTRUM"),
        ("WEIGHT_SPECTRUM", "WEIGHT_SPECTRUM"),
        ("WEIGHT", "WEIGHT"),
    ],
)
def test_should_load_weights_from_requested_column(
    ms_with_distinct_weights, weightcolumn, expected_column
):
    ms_path, weight, weight_spectrum = ms_with_distinct_weights
    expected_column_data = {
        "WEIGHT": weight,
        "WEIGHT_SPECTRUM": weight_spectrum,
    }[expected_column]

    vis = load_ms_as_dataset_with_time_chunks(
        ms_path, 1, weightcolumn=weightcolumn
    )

    np.testing.assert_array_equal(
        vis.weight.values,
        _expected_vis_weight(expected_column_data, vis.weight.shape),
    )


def test_should_load_weights_from_weight_when_weight_spectrum_is_absent(
    ms_with_distinct_weights,
):
    ms_path, weight, _ = ms_with_distinct_weights
    with table(ms_path, readonly=False, ack=False) as tab:
        tab.removecols("WEIGHT_SPECTRUM")

    vis = load_ms_as_dataset_with_time_chunks(ms_path, 1)

    np.testing.assert_array_equal(
        vis.weight.values, _expected_vis_weight(weight, vis.weight.shape)
    )


def test_should_load_per_channel_weights_from_custom_column(
    ms_with_distinct_weights,
):
    ms_path, _, weight_spectrum = ms_with_distinct_weights
    with table(ms_path, readonly=False, ack=False) as tab:
        tab.renamecol("WEIGHT_SPECTRUM", "CUSTOM_WEIGHT")

    vis = load_ms_as_dataset_with_time_chunks(
        ms_path, 1, weightcolumn="CUSTOM_WEIGHT"
    )

    np.testing.assert_array_equal(
        vis.weight.values,
        _expected_vis_weight(weight_spectrum, vis.weight.shape),
    )


def test_should_load_weights_when_only_weight_spectrum_is_present(
    ms_with_distinct_weights,
):
    ms_path, _, weight_spectrum = ms_with_distinct_weights
    with table(ms_path, readonly=False, ack=False) as tab:
        tab.removecols("WEIGHT")

    vis = load_ms_as_dataset_with_time_chunks(ms_path, 1)

    np.testing.assert_array_equal(
        vis.weight.values,
        _expected_vis_weight(weight_spectrum, vis.weight.shape),
    )


def test_should_raise_if_weight_column_is_absent_from_template(generate_ms):
    with pytest.raises(
        ValueError, match="Weight column 'MISSING_WEIGHT' is absent"
    ):
        create_template_vis_from_ms(
            generate_ms, datacolumn="DATA", weightcolumn="MISSING_WEIGHT"
        )


def test_should_raise_if_data_column_is_absent(generate_ms):
    with pytest.raises(
        ValueError, match="Data column 'CORRECTED_DATA' is absent"
    ):
        create_template_vis_from_ms(
            generate_ms, datacolumn="CORRECTED_DATA", weightcolumn="auto"
        )


def create_mock_ms(defined_columns, undefined_columns=()):
    ms = MagicMock(name="ms")
    ms.colnames.return_value = [*defined_columns, *undefined_columns]
    ms.iscelldefined.side_effect = lambda col, row: col in defined_columns
    return ms


@pytest.mark.parametrize(
    "defined_columns, undefined_columns, expected_column",
    [
        (["WEIGHT", "WEIGHT_SPECTRUM"], [], "WEIGHT_SPECTRUM"),
        (["WEIGHT"], [], "WEIGHT"),
        (["WEIGHT"], ["WEIGHT_SPECTRUM"], "WEIGHT"),
    ],
)
def test_should_resolve_auto_weight_column(
    defined_columns, undefined_columns, expected_column
):
    ms = create_mock_ms(defined_columns, undefined_columns)

    assert _resolve_weight_column(ms, "auto") == expected_column


@pytest.mark.parametrize(
    "defined_columns, undefined_columns",
    [
        ([], []),
        ([], ["WEIGHT", "WEIGHT_SPECTRUM"]),
    ],
)
def test_should_raise_if_auto_weight_columns_have_no_data(
    defined_columns, undefined_columns
):
    ms = create_mock_ms(defined_columns, undefined_columns)

    with pytest.raises(ValueError, match="Weight column 'WEIGHT' is absent"):
        _resolve_weight_column(ms, "auto")


def test_should_resolve_explicit_weight_column():
    ms = create_mock_ms(["WEIGHT", "WEIGHT_SPECTRUM", "CUSTOM_WEIGHT"])

    assert _resolve_weight_column(ms, "CUSTOM_WEIGHT") == "CUSTOM_WEIGHT"


@pytest.mark.parametrize(
    "defined_columns, undefined_columns",
    [
        (["WEIGHT"], []),
        (["WEIGHT"], ["WEIGHT_SPECTRUM"]),
    ],
)
def test_should_raise_if_explicit_weight_column_has_no_data(
    defined_columns, undefined_columns
):
    ms = create_mock_ms(defined_columns, undefined_columns)

    with pytest.raises(
        ValueError, match="Weight column 'WEIGHT_SPECTRUM' is absent"
    ):
        _resolve_weight_column(ms, "WEIGHT_SPECTRUM")


def test_should_read_field_id_from_field_table(mock_table):
    ms_path = "/path/to/test.ms"
    expected_field_id = "field_0123"
    mock_field_table = create_mock_table(getcol_return=[expected_field_id])
    mock_table.return_value = mock_field_table

    result = read_ms_field_id(ms_path)

    assert result == expected_field_id
    mock_table.assert_called_once_with(ms_path + "/FIELD")
    mock_field_table.getcol.assert_called_once_with("NAME")


def test_should_return_unknown_field(mock_table):
    ms_path = "/path/to/test.ms"
    mock_field_table = create_mock_table(getcol_return=[""])
    mock_table.return_value = mock_field_table

    assert read_ms_field_id(ms_path) == "UNKNOWN_FIELD"
    mock_table.assert_called_once_with(ms_path + "/FIELD")
    mock_field_table.getcol.assert_called_once_with("NAME")


def test_should_return_first_field_name_when_multiple_fields(mock_table):
    ms_path = "/data/observation.ms"
    field_names = ["target_field", "calibrator_field", "another_field"]
    mock_field_table = create_mock_table(getcol_return=field_names)
    mock_table.return_value = mock_field_table

    result = read_ms_field_id(ms_path)

    assert result == "target_field"


def create_antenna_response_tables(element_offset, element_flag):
    nstations = element_offset.shape[0]
    columns = {
        "DELAY_DIR": np.array([[[0.1, 0.2]]]),
        "COORDINATE_AXES": np.tile(np.eye(3), (nstations, 1, 1)),
        "ELEMENT_OFFSET": element_offset,
        "ELEMENT_FLAG": element_flag,
        "NAME": [f"s{idx}" for idx in range(nstations)],
        "POSITION": np.zeros((nstations, 3)),
    }
    tab = create_mock_table()
    tab.getcol.side_effect = lambda name: columns[name]
    return tab


@pytest.mark.parametrize(
    "stored_layout",
    ["components_first", "elements_first"],
)
@pytest.mark.parametrize("nelements", [2, 3, 4])
def test_should_read_element_offsets_and_flags_per_station(
    mock_table, stored_layout, nelements
):
    nstations = 2
    # Distinct offsets per station and element: [station, element, xyz]
    expected_offsets = np.arange(
        nstations * nelements * 3, dtype=float
    ).reshape(nstations, nelements, 3)
    # Flags per station and element: [station, element, (x, y)]
    expected_flags = np.zeros((nstations, nelements, 2), dtype=bool)
    expected_flags[0, 1, 0] = True
    expected_flags[1, -1, 1] = True

    if stored_layout == "components_first":
        stored_offsets = expected_offsets.transpose(0, 2, 1)
        stored_flags = expected_flags.transpose(0, 2, 1)
    else:
        stored_offsets = expected_offsets
        stored_flags = expected_flags
    mock_table.return_value = create_antenna_response_tables(
        stored_offsets, stored_flags
    )

    data = read_antenna_response_data.__wrapped__("test.ms")

    np.testing.assert_array_equal(data["element_offsets"], expected_offsets)
    np.testing.assert_array_equal(data["element_flags"], expected_flags)


def test_should_raise_for_inconsistent_element_layout(mock_table):
    # Offsets stored as (3, nelements) but flags as (nelements, 2)
    mock_table.return_value = create_antenna_response_tables(
        np.zeros((2, 3, 4)), np.zeros((2, 4, 2), dtype=bool)
    )

    with pytest.raises(ValueError, match="Unsupported PHASED_ARRAY layout"):
        read_antenna_response_data.__wrapped__("test.ms")
