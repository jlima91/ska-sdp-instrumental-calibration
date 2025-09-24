import numpy as np

from ska_sdp_instrumental_calibration.processing_tasks.gain_flagging import (
    GainFlagger,
)


def test_should_flag_gains():

    soltype = "amplitude"
    mode = "smooth"
    order = 3
    max_rms = 5.0
    fix_rms = 0.0
    max_ncycles = 1
    max_rms_noise = 0.0
    window_noise = 3
    fix_rms_noise = 0.0
    frequencies = np.arange(0, 1, 0.1)
    gains = np.arange(1, 2, 0.1) + 1j * np.arange(2, 1, -0.1)
    gains[5] = 0 + 100j
    weights = np.ones(10)

    flagger_obj = GainFlagger(
        soltype,
        mode,
        order,
        max_rms,
        fix_rms,
        max_ncycles,
        max_rms_noise,
        window_noise,
        fix_rms_noise,
        frequencies,
    )
    updated_weights = flagger_obj.flag_dimension(
        gains, weights, "a1", "X", "Y"
    )
    print(np.angle(gains))
    print(np.absolute(gains))
    expected = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    np.testing.assert_allclose(updated_weights, expected)
