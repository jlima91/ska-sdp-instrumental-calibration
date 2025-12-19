import numpy as np

from ska_sdp_instrumental_calibration.numpy_processors.gaintable import (
    apply_antenna_gains_to_visibility,
)


def test_should_apply_gaintable_to_visibility():
    vis = np.arange(48).reshape(2, 2, 3, 4)
    gains = np.arange(24).reshape(1, 2, 3, 2, 2)
    antenna1 = np.arange(2)
    antenna2 = np.arange(2)

    updated_vis = apply_antenna_gains_to_visibility(
        vis, gains, antenna1, antenna2
    )

    expected = np.array(
        [
            [
                [
                    [3, 13, 11, 45],
                    [459, 661, 659, 949],
                    [2771, 3421, 3419, 4221],
                ],
                [
                    [8475, 9829, 9827, 11397],
                    [19107, 21421, 21419, 24013],
                    [36203, 39733, 39731, 43605],
                ],
            ],
            [
                [
                    [27, 133, 131, 645],
                    [2403, 3469, 3467, 5005],
                    [9707, 11989, 11987, 14805],
                ],
                [
                    [23475, 27229, 27227, 31581],
                    [45243, 50725, 50723, 56869],
                    [76547, 84013, 84011, 92205],
                ],
            ],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(updated_vis, expected)


def test_should_apply_gaintable_to_visibility_inverted():
    vis = np.array(
        [
            [
                [
                    [3, 13, 11, 45],
                    [459, 661, 659, 949],
                    [2771, 3421, 3419, 4221],
                ],
                [
                    [8475, 9829, 9827, 11397],
                    [19107, 21421, 21419, 24013],
                    [36203, 39733, 39731, 43605],
                ],
            ],
            [
                [
                    [27, 133, 131, 645],
                    [2403, 3469, 3467, 5005],
                    [9707, 11989, 11987, 14805],
                ],
                [
                    [23475, 27229, 27227, 31581],
                    [45243, 50725, 50723, 56869],
                    [76547, 84013, 84011, 92205],
                ],
            ],
        ],
        dtype=int,
    )
    gains = np.arange(24).reshape(1, 2, 3, 2, 2)
    antenna1 = np.arange(2)
    antenna2 = np.arange(2)

    updated_vis = apply_antenna_gains_to_visibility(
        vis, gains, antenna1, antenna2, inverse=True
    )

    expected = np.arange(48, dtype=float).reshape(2, 2, 3, 4)

    np.testing.assert_allclose(updated_vis, expected)
