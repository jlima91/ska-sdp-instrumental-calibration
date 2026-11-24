import numpy as np
import pytest

from ska_sdp_instrumental_calibration.data_managers.schema.calibration_set import (  # noqa: E501
    GainCalibrationSetXds,
)
from ska_sdp_instrumental_calibration.qa_metrics.bandpass_fit_metric import (
    SKA_LOW_COARSE_CHANNEL_WIDTH,
    bandpass_fit_quality,
    coarse_channel_amplitude_residuals,
    count_good_phase_channels,
    count_within_n_sigma,
    gain_fit_quality_for_pol,
    phase_fit_quality_for_pol,
)

POLARIZATIONS = ["XX", "XY", "YX", "YY"]
N_COARSE = 16
N_FINE = 24
# Channels of N_COARSE coarse channels, N_FINE per coarse channel, centred
# on the coarse channel centres (n * SKA_LOW_COARSE_CHANNEL_WIDTH)
FREQUENCY = np.concatenate(
    [
        n * SKA_LOW_COARSE_CHANNEL_WIDTH
        + (np.arange(N_FINE) - (N_FINE - 1) / 2)
        * SKA_LOW_COARSE_CHANNEL_WIDTH
        / N_FINE
        for n in range(160, 160 + N_COARSE)
    ]
)
# Channels made outliers: one in every 4th coarse channel, away from its
# edges. The fit is a single least squares pass, so outliers pull it: few
# enough of them keep the good channels within n_sigma of it.
OUTLIERS = np.arange(0, N_COARSE, 4) * N_FINE + 5


def _spectrum(seed=0):
    """
    Gain spectrum with a smooth amplitude and a delay + TEC phase, plus a
    little noise
    """
    rng = np.random.default_rng(seed)
    amplitude = (
        1
        + 0.3 * np.sin(FREQUENCY / 5e6)
        + rng.normal(0, 0.005, FREQUENCY.size)
    )
    phase = (
        0.3
        + 2 * np.pi * FREQUENCY * 20e-9
        + 2e8 / FREQUENCY
        + rng.normal(0, 0.02, FREQUENCY.size)
    )
    return amplitude * np.exp(1j * phase)


def _with_outliers(gains, amplitude=1.0, phase=0.0):
    """
    Gains with the OUTLIERS channels scaled and rotated. An amplitude
    scale of 1.06 is >10 sigma of the amplitude noise, yet small enough
    not to pull the line of its coarse channel (24 channels) away from its
    good channels.
    """
    gains = gains.copy()
    gains[OUTLIERS] *= amplitude * np.exp(1j * phase)
    return gains


def _gaintable(spectra, frequency=FREQUENCY, weight=None):
    """
    Gaintable of one time from the XX and YY spectrum of each station

    Parameters
    ----------
    spectra : dict[str, tuple[ndarray, ndarray]]
        XX and YY gains per station name.
    """
    station_names = list(spectra)
    shape = (1, len(station_names), frequency.size, len(POLARIZATIONS))
    gain = np.zeros(shape, dtype=complex)
    for i, (xx, yy) in enumerate(spectra.values()):
        gain[0, i, :, 0] = xx
        gain[0, i, :, 3] = yy
    return GainCalibrationSetXds.constructor(
        gain=gain,
        weight=np.ones(shape) if weight is None else weight,
        residual=np.zeros((1, frequency.size, len(POLARIZATIONS))),
        time=np.array([10.0]),
        interval=np.array([np.nan]),
        frequency=frequency,
        antenna_name=np.array(station_names),
        polarization=np.array(POLARIZATIONS),
        jones_type="B",
    )


def _station(gaintable, station_name="station_a"):
    return gaintable.sel(antenna_name=station_name)


def test_should_count_residuals_within_n_sigma():
    # IQR of the 100 residuals is 0.5, so sigma = 0.5 / 1.349: 5 sigma
    # is ~1.85; only the 10.0 is beyond it
    residuals = np.concatenate([np.linspace(-0.5, 0.5, 99), [10.0]])

    assert count_within_n_sigma(residuals, n_sigma=5.0) == 99


def test_should_count_all_residuals_of_a_perfect_fit():
    # sigma is 0: no channel is beyond the fit
    assert count_within_n_sigma(np.zeros(10), n_sigma=5.0) == 10


def test_should_count_all_phase_channels_of_a_delay_and_tec_phase():
    phase = np.unwrap(np.angle(_spectrum()))

    assert (
        count_good_phase_channels(FREQUENCY, phase, n_sigma=5.0)
        == FREQUENCY.size
    )


@pytest.mark.xfail(
    strict=True,
    reason="Without noise the residuals are rounding errors (~1e-14 rad), "
    "so the robust sigma is too and most channels are beyond n_sigma of "
    "the fit: sigma needs a floor",
)
def test_should_count_all_phase_channels_of_a_noiseless_phase():
    nu = FREQUENCY
    phase = 0.3 + 2 * np.pi * nu * 20e-9 + 2e8 / nu

    assert count_good_phase_channels(nu, phase, n_sigma=5.0) == nu.size


def test_should_not_count_phase_channels_far_from_the_fit():
    gains = _with_outliers(_spectrum(), phase=1.5)
    phase = np.unwrap(np.angle(gains))

    assert count_good_phase_channels(
        FREQUENCY, phase, n_sigma=5.0
    ) == FREQUENCY.size - len(OUTLIERS)


def test_should_fit_a_line_per_coarse_channel():
    # A different line in each coarse channel: each is fitted exactly
    coarse = np.round(FREQUENCY / SKA_LOW_COARSE_CHANNEL_WIDTH).astype(int)
    amplitude = coarse + 1e-6 * coarse * (FREQUENCY - coarse * 781250.0)

    residuals = coarse_channel_amplitude_residuals(
        FREQUENCY, amplitude, coarse
    )

    assert residuals.size == FREQUENCY.size
    np.testing.assert_allclose(residuals, 0, atol=1e-9)


def test_should_match_a_polynomial_fit_of_each_coarse_channel():
    amplitude = np.abs(_spectrum())
    coarse = np.round(FREQUENCY / SKA_LOW_COARSE_CHANNEL_WIDTH).astype(int)

    residuals = coarse_channel_amplitude_residuals(
        FREQUENCY, amplitude, coarse
    )

    expected = np.concatenate(
        [
            amplitude[coarse == c]
            - np.polyval(
                np.polyfit(FREQUENCY[coarse == c], amplitude[coarse == c], 1),
                FREQUENCY[coarse == c],
            )
            for c in np.unique(coarse)
        ]
    )
    np.testing.assert_allclose(residuals, expected, atol=1e-9)


def test_should_not_fit_coarse_channels_with_too_few_channels():
    # One coarse channel of 2 channels, one of 3
    nu = np.array([0.0, 1.0, 1e7, 1e7 + 1, 1e7 + 2])
    coarse = np.array([0, 0, 1, 1, 1])

    residuals = coarse_channel_amplitude_residuals(nu, np.ones(5), coarse)

    assert residuals.size == 3


def test_should_get_no_residuals_if_no_coarse_channel_can_be_fitted():
    residuals = coarse_channel_amplitude_residuals(
        np.array([0.0, 1e7]), np.ones(2), np.array([0, 1])
    )

    assert residuals.size == 0


@pytest.mark.parametrize(
    "fit_quality_for_pol, outliers",
    [
        (phase_fit_quality_for_pol, {"phase": 1.5}),
        (gain_fit_quality_for_pol, {"amplitude": 1.06}),
    ],
)
def test_should_get_ratio_of_good_channels_of_a_polarisation(
    fit_quality_for_pol, outliers
):
    gaintable = _gaintable(
        {"station_a": (_spectrum(), _with_outliers(_spectrum(), **outliers))}
    )

    assert fit_quality_for_pol(_station(gaintable), "XX") == 1.0
    assert fit_quality_for_pol(_station(gaintable), "YY") == pytest.approx(
        1 - len(OUTLIERS) / FREQUENCY.size
    )


@pytest.mark.parametrize(
    "fit_quality_for_pol",
    [phase_fit_quality_for_pol, gain_fit_quality_for_pol],
)
def test_should_leave_flagged_channels_out_of_the_fit_quality(
    fit_quality_for_pol,
):
    # The outliers are flagged: the remaining channels are all good
    weight = np.ones((1, 1, FREQUENCY.size, len(POLARIZATIONS)))
    weight[0, 0, OUTLIERS, :] = 0
    gains = _with_outliers(_spectrum(), amplitude=1.2, phase=1.5)
    gaintable = _gaintable({"station_a": (gains, gains)}, weight=weight)

    assert fit_quality_for_pol(_station(gaintable), "XX") == 1.0


@pytest.mark.parametrize(
    "fit_quality_for_pol",
    [phase_fit_quality_for_pol, gain_fit_quality_for_pol],
)
def test_should_get_no_fit_quality_if_all_channels_are_flagged(
    fit_quality_for_pol,
):
    gaintable = _gaintable(
        {"station_a": (_spectrum(), _spectrum())},
        weight=np.zeros((1, 1, FREQUENCY.size, len(POLARIZATIONS))),
    )

    assert fit_quality_for_pol(_station(gaintable), "XX") is None


def test_should_get_no_gain_fit_quality_at_coarse_channel_resolution():
    # One channel per coarse channel: no coarse channel can be fitted,
    # but the phase across the band still can
    frequency = np.arange(160, 160 + N_COARSE) * SKA_LOW_COARSE_CHANNEL_WIDTH
    rng = np.random.default_rng(0)
    gains = np.exp(
        1j
        * (2 * np.pi * frequency * 20e-9 + rng.normal(0, 0.02, frequency.size))
    )
    gaintable = _gaintable({"station_a": (gains, gains)}, frequency=frequency)

    assert gain_fit_quality_for_pol(_station(gaintable), "XX") is None
    assert phase_fit_quality_for_pol(_station(gaintable), "XX") == 1.0


def test_should_get_fit_quality_of_each_station():
    bad = _with_outliers(_spectrum(), amplitude=1.06, phase=1.5)
    gaintable = _gaintable(
        {
            "station_a": (_spectrum(), bad),
            "station_b": (bad, _spectrum(seed=1)),
        }
    )
    expected_bad = pytest.approx(1 - len(OUTLIERS) / FREQUENCY.size)

    assert bandpass_fit_quality(gaintable) == {
        "station_a": {
            "x_phase_fit_quality": 1.0,
            "x_gains_fit_quality": 1.0,
            "y_phase_fit_quality": expected_bad,
            "y_gains_fit_quality": expected_bad,
        },
        "station_b": {
            "x_phase_fit_quality": expected_bad,
            "x_gains_fit_quality": expected_bad,
            "y_phase_fit_quality": 1.0,
            "y_gains_fit_quality": 1.0,
        },
    }


def test_should_get_same_fit_quality_in_any_frequency_order():
    # Unwrapping the phase needs ascending frequency: a gaintable in
    # descending order is sorted first
    bad = _with_outliers(_spectrum(), amplitude=1.06, phase=1.5)
    gaintable = _gaintable({"station_a": (_spectrum(), bad)})

    assert bandpass_fit_quality(
        gaintable.isel(frequency=slice(None, None, -1))
    ) == bandpass_fit_quality(gaintable)
