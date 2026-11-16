import dask.array as da
import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.schema.polarization import (  # noqa: E501
    jones_to_pol,
    npol_to_nrec,
    pol_index,
    pol_to_jones,
)


@pytest.mark.parametrize("npol, nrec", [(1, 1), (4, 2)])
def test_npol_to_nrec(npol, nrec):
    assert npol_to_nrec(npol) == nrec


@pytest.mark.parametrize("npol", [0, 2, 3, 5])
def test_npol_to_nrec_should_raise_for_unsupported_npol(npol):
    with pytest.raises(
        ValueError, match=f"Unsupported number of polarizations: {npol}"
    ):
        npol_to_nrec(npol)


@pytest.mark.parametrize(
    "rec1, rec2, nrec, expected",
    [(0, 0, 2, 0), (0, 1, 2, 1), (1, 0, 2, 2), (1, 1, 2, 3), (0, 0, 1, 0)],
)
def test_pol_index(rec1, rec2, nrec, expected):
    assert pol_index(rec1, rec2, nrec) == expected


def test_pol_index_should_match_pol_to_jones():
    pols = np.arange(4)
    jones = pol_to_jones(pols)

    for rec1, rec2 in np.ndindex(2, 2):
        assert pols[pol_index(rec1, rec2)] == jones[rec1, rec2]


@pytest.mark.parametrize(
    "rec1, rec2, nrec", [(2, 0, 2), (0, 2, 2), (-1, 0, 2), (0, 1, 1)]
)
def test_pol_index_should_raise_for_out_of_range_receptors(rec1, rec2, nrec):
    with pytest.raises(ValueError, match="out of range"):
        pol_index(rec1, rec2, nrec)


def test_pol_to_jones_should_reshape_in_row_major_order():
    x = np.arange(3 * 4).reshape(3, 4)

    jones = pol_to_jones(x)

    assert jones.shape == (3, 2, 2)
    np.testing.assert_array_equal(jones[1], [[4, 5], [6, 7]])


def test_pol_to_jones_should_support_scalar_and_dask():
    x = da.ones((3, 1))

    jones = pol_to_jones(x)

    assert isinstance(jones, da.Array)
    assert jones.shape == (3, 1, 1)


def test_pol_to_jones_should_raise_for_two_pols():
    with pytest.raises(
        ValueError, match="Unsupported number of polarizations: 2"
    ):
        pol_to_jones(np.zeros((3, 2)))


def test_jones_to_pol_should_invert_pol_to_jones():
    x = np.arange(5 * 4).reshape(5, 4)

    np.testing.assert_array_equal(jones_to_pol(pol_to_jones(x)), x)
