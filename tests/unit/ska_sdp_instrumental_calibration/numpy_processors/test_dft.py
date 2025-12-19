import numpy as np
from mock import MagicMock

from ska_sdp_instrumental_calibration.numpy_processors.dft import (
    dft_skycomponent,
)


def test_should_perform_dft():
    sky_comp = MagicMock(name="sky_comp")
    sky_comp.frequency = np.arange(5)
    sky_comp.direction.dec.radian = np.pi / 4
    sky_comp.direction.ra.radian = np.pi / 5
    sky_comp.shape = "GAUSSIAN"
    sky_comp.params = {"bpa": 1.0, "bmaj": 1.1, "bmin": 0.1}
    sky_comp.flux = np.arange(10).reshape(5, 2)
    sky_coord = MagicMock(name="sky_coord")
    sky_coord.ra.radian = np.pi / 4
    sky_coord.dec.radian = np.pi / 4

    uvw = np.arange(18).reshape(2, 3, 3)

    actual = dft_skycomponent(uvw, sky_comp, sky_coord)
    expected = [
        [
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 2.58033301e-10j, 3.0 + 3.87049951e-10j],
                [4.0 + 1.03213320e-09j, 5.0 + 1.29016650e-09j],
                [6.0 + 2.32229971e-09j, 7.0 + 2.70934966e-09j],
                [8.0 + 4.12853281e-09j, 9.0 + 4.64459941e-09j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 1.41680567e-08j, 3.0 + 2.12520850e-08j],
                [4.0 + 5.66722266e-08j, 5.0 + 7.08402833e-08j],
                [6.0 + 1.27512510e-07j, 7.0 + 1.48764595e-07j],
                [8.0 + 2.26688907e-07j, 9.0 + 2.55025020e-07j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 2.80780800e-08j, 3.0 + 4.21171200e-08j],
                [4.0 + 1.12312320e-07j, 5.0 + 1.40390400e-07j],
                [6.0 + 2.52702720e-07j, 7.0 + 2.94819840e-07j],
                [8.0 + 4.49249280e-07j, 9.0 + 5.05405440e-07j],
            ],
        ],
        [
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 4.19881034e-08j, 3.0 + 6.29821551e-08j],
                [4.0 + 1.67952414e-07j, 5.0 + 2.09940517e-07j],
                [6.0 + 3.77892930e-07j, 7.0 + 4.40875086e-07j],
                [8.0 + 6.71809654e-07j, 9.0 + 7.55785861e-07j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 5.58981267e-08j, 3.0 + 8.38471901e-08j],
                [4.0 + 2.23592507e-07j, 5.0 + 2.79490634e-07j],
                [6.0 + 5.03083141e-07j, 7.0 + 5.86930331e-07j],
                [8.0 + 8.94370028e-07j, 9.0 + 1.00616628e-06j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 6.98081501e-08j, 3.0 + 1.04712225e-07j],
                [4.0 + 2.79232600e-07j, 5.0 + 3.49040751e-07j],
                [6.0 + 6.28273351e-07j, 7.0 + 7.32985576e-07j],
                [8.0 + 1.11693040e-06j, 9.0 + 1.25654670e-06j],
            ],
        ],
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)


def test_should_perform_dft_without_gaussian_tappers():
    sky_comp = MagicMock(name="sky_comp")
    sky_comp.frequency = np.arange(5)
    sky_comp.direction.dec.radian = np.pi / 4
    sky_comp.direction.ra.radian = np.pi / 5
    sky_comp.shape = "NON_GAUSSIAN"
    sky_comp.params = {"bpa": 1.0, "bmaj": 1.1, "bmin": 0.1}
    sky_comp.flux = np.arange(10).reshape(5, 2)
    sky_coord = MagicMock(name="sky_coord")
    sky_coord.ra.radian = np.pi / 4
    sky_coord.dec.radian = np.pi / 4

    uvw = np.arange(18).reshape(2, 3, 3)

    actual = dft_skycomponent(uvw, sky_comp, sky_coord)
    expected = [
        [
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 2.58033301e-10j, 3.0 + 3.87049951e-10j],
                [4.0 + 1.03213320e-09j, 5.0 + 1.29016650e-09j],
                [6.0 + 2.32229971e-09j, 7.0 + 2.70934966e-09j],
                [8.0 + 4.12853281e-09j, 9.0 + 4.64459941e-09j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 1.41680567e-08j, 3.0 + 2.12520850e-08j],
                [4.0 + 5.66722266e-08j, 5.0 + 7.08402833e-08j],
                [6.0 + 1.27512510e-07j, 7.0 + 1.48764595e-07j],
                [8.0 + 2.26688907e-07j, 9.0 + 2.55025020e-07j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 2.80780800e-08j, 3.0 + 4.21171200e-08j],
                [4.0 + 1.12312320e-07j, 5.0 + 1.40390400e-07j],
                [6.0 + 2.52702720e-07j, 7.0 + 2.94819840e-07j],
                [8.0 + 4.49249280e-07j, 9.0 + 5.05405440e-07j],
            ],
        ],
        [
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 4.19881034e-08j, 3.0 + 6.29821551e-08j],
                [4.0 + 1.67952414e-07j, 5.0 + 2.09940517e-07j],
                [6.0 + 3.77892930e-07j, 7.0 + 4.40875086e-07j],
                [8.0 + 6.71809654e-07j, 9.0 + 7.55785861e-07j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 5.58981267e-08j, 3.0 + 8.38471901e-08j],
                [4.0 + 2.23592507e-07j, 5.0 + 2.79490634e-07j],
                [6.0 + 5.03083141e-07j, 7.0 + 5.86930331e-07j],
                [8.0 + 8.94370028e-07j, 9.0 + 1.00616628e-06j],
            ],
            [
                [0.0 + 0.00000000e00j, 1.0 + 0.00000000e00j],
                [2.0 + 6.98081501e-08j, 3.0 + 1.04712225e-07j],
                [4.0 + 2.79232600e-07j, 5.0 + 3.49040751e-07j],
                [6.0 + 6.28273351e-07j, 7.0 + 7.32985576e-07j],
                [8.0 + 1.11693040e-06j, 9.0 + 1.25654670e-06j],
            ],
        ],
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=0)
