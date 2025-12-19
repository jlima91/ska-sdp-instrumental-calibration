import numpy as np

from ska_sdp_instrumental_calibration.numpy_processors.rotation_matrix import (
    generate_rotation_matrices,
)


def test_should_generate_rotation_matrices():
    rm = np.arange(2)
    frequency = (np.arange(3) + 0.5) * 1e5

    actual = generate_rotation_matrices(rm, frequency)
    expected = np.array(
        [
            [
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            ],
            [
                [
                    [0.4650611 + 0.0j, -0.8852786 + 0.0j],
                    [0.8852786 + 0.0j, 0.4650611 + 0.0j],
                ],
                [
                    [0.05371898 + 0.0j, -0.9985561 + 0.0j],
                    [0.9985561 + 0.0j, 0.05371898 + 0.0j],
                ],
                [
                    [0.69852227 + 0.0j, -0.7155883 + 0.0j],
                    [0.7155883 + 0.0j, 0.69852227 + 0.0j],
                ],
            ],
        ]
    )

    np.testing.assert_allclose(actual, expected)
