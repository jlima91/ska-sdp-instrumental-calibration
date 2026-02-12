import logging

import dask
import numpy as np
import xarray as xr
from scipy.ndimage import generic_filter
from scipy.optimize import curve_fit

logger = logging.getLogger()


@dask.delayed
def log_flaging_statistics(weights, initial_weights):
    current_flagged = (
        weights[:, :, :, 0, 0] != initial_weights[:, :, :, 0, 0]
    ).sum(dim=["time", "frequency"])

    antna_percent_flagged = (
        current_flagged / weights[:, 0, :, 0, 0].size
    ) * 100

    min_percent = antna_percent_flagged.min()
    median_percent = np.median(antna_percent_flagged.data)
    max_percent = antna_percent_flagged.max()

    logger.info(
        f"Gain flagging: Statistics "
        f" min: {min_percent.data:.2f}%,"
        f" median: {median_percent:.2f}%,"
        f" max: {max_percent.data:.2f}%."
    )


class SmoothingFit:
    def __init__(self, order, component):
        """
        Performs Smooth fit on the gains.

        Parameters
        ----------
            order: int
                Order/size of the fitter.
            component: dict
                soltype function.
        """
        self.order = order
        self.component = component

    def fit(self, gains, weights):
        """
        Fits a smooth curve on the gains.

        Parameters
        ----------
            gains: Array
                Gains.
            weights: Array
                Weights of the gains.
        Returns
        -------
            Dict of fits.
        """
        fits = {}
        components = self.component(gains)

        for name, arr in components.items():
            vals = np.copy(arr)
            np.putmask(vals, weights == 0, np.nan)

            fits[name] = generic_filter(
                vals,
                np.nanmedian,
                size=self.order,
                mode="constant",
                cval=np.nan,
            )

        return fits


class PhasorPolyFit:
    def __init__(self, order, freq):
        """
        Performs poly fit on the gains.

        Parameters
        ----------
            order: int
                Order/size of the fitter.
            freq: Array
                Frequency.
        """
        self.order = order
        self.freq = freq

    def fit(self, gains, weights, freq_guess):
        """
        Fit a phasor model with polynomial envelope.

        Parameters
        ----------
            gains: Array
                Gains.
            weights: Array
                Weights of the gains.
            freq_guess: float
                Initial frequency to begin.

        Returns
        -------
            Fitted model.
        """
        mask = weights != 0
        if mask.sum() < self.order + 2:
            return np.full_like(gains, np.nan, dtype=complex), freq_guess

        x = self.freq[mask]
        y = gains[mask]

        if freq_guess is None:
            dx = np.median(np.diff(x))
            fft_freqs = np.fft.fftfreq(len(x), dx)
            fft_power = np.abs(np.fft.fft(y - y.mean()))
            freq_guess = np.abs(fft_freqs[fft_power.argmax()])

        env_mean = np.mean(y * np.exp(-1j * 2 * np.pi * freq_guess * x))
        p0 = [freq_guess, env_mean.real, env_mean.imag]

        for _ in range(self.order):
            p0.extend([0.0, 0.0])

        y_real = np.concatenate([y.real, y.imag])

        try:
            params, _ = curve_fit(
                self.real_wrapper,
                x,
                y_real,
                p0=p0,
                maxfev=5000,
            )
            model = self.phasor_envelope_model(self.freq, *params)
            return model, freq_guess

        except RuntimeError:
            logger.warning("Phasor fit failed, returning NaNs")
            return np.full_like(gains, np.nan, dtype=complex), freq_guess

    def phasor_envelope_model(self, x_arr, freq, *params):
        """
        Phasor model with polynomial envelope.

        Parameters
        ----------
            x_arr: Array
                gains
            freq: Array
                Frequency
            *params: (tuple[float])
                Polynomial coefficients.
        Returns
        -------
            Phasor model array.
        """
        n_coeffs = len(params) // 2
        envelope = np.zeros_like(x_arr, dtype=complex)

        for n in range(n_coeffs):
            coeff = params[2 * n] + 1j * params[2 * n + 1]
            envelope += coeff * x_arr**n

        return envelope * np.exp(1j * 2 * np.pi * freq * x_arr)

    def real_wrapper(self, x_arr, *params):
        """
        Wrapper for curve_fit to handle complex y_arr.

        Parameters
        ----------
            x_arr: Array
                gains
            *params: (tuple[float])
                Fitting parameters.
        Returns
        -------
            A complex array disguised as a real-valued one.
        """
        result = self.phasor_envelope_model(x_arr, *params)
        return np.concatenate([result.real, result.imag])


class RMSFlagger:
    def __init__(self, n_sigma):
        """
        Performs RMS flagging.

        Parameters
        ----------
            n_sigma: float
                Flag values greated than n_simga * sigma_hat.
                Where sigma_hat is 1.4826 * MeanAbsoluteDeviation
        """
        self.n_sigma = n_sigma

    def flag(self, detrended, weights):
        """
        Does flagging using rms.

        Parameters
        ----------
            deterend: Array
                Diff of fit and gains.
            weights: Array
                Weights of gains.
        Returns
        -------
            Array fo flags
        """
        valid = weights != 0
        if not np.any(valid):
            return np.zeros_like(weights, dtype=bool), np.nan

        sigma = 1.4826 * np.nanmedian(np.abs(detrended[valid]))
        flags = np.abs(detrended) > self.n_sigma * sigma
        return flags, sigma


class RollingRMSFlagger:
    def __init__(self, n_sigma, window):
        if window % 2 != 1:
            raise ValueError("window_size must be odd")
        self.n_sigma = n_sigma
        self.window = window

    def flag(self, detrended, weights):
        """
        Does flagging using rolling rms.

        Parameters
        ----------
            deterend: Array
                Diff of fit and gains.
            weights: Array
                Weights of gains.
        Returns
        -------
            Array fo flags
        """
        valid = weights != 0
        pad = np.pad(detrended, self.window // 2, mode="reflect")
        rms = np.sqrt(
            np.convolve(pad**2, np.ones(self.window), "valid") / self.window
        )

        sigma = 1.4826 * np.nanmedian(np.abs(rms[valid]))
        flags = rms > self.n_sigma * sigma
        return flags, sigma


class GainFlagger:
    MODE = {
        "poly": {"amplitude", "phase", "amp-phase"},
        "smooth": {"real-imag"},
    }

    SOLTYPE = {
        "amplitude": lambda a: {"amp_fit": np.abs(a)},
        "phase": lambda a: {"phase_fit": np.angle(a)},
        "amp-phase": lambda a: {
            "amp_fit": np.abs(a),
            "phase_fit": np.angle(a),
        },
        "real-imag": lambda a: {
            "real_fit": a.real,
            "imag_fit": a.imag,
        },
    }

    def __init__(
        self,
        soltype: str,
        mode: str,
        order: int,
        max_ncycles: int,
        n_sigma: float,
        n_sigma_rolling: float,
        window_size: int,
        freq: np.ndarray,
    ):
        """
        Generates gain flagger for given soltype and fitting parameters.

        Parameters
        ----------
            soltype: str
                Solution type to flag. Can be "phase", "amplitude"
                or "both".
            mode: str
                Detrending/fitting algorithm: "smooth", "poly".
                By default smooth.
            order : int
                Order of the function fitted during detrending.
                If mode=smooth these are the window of the running
                median (0=all axis).
            max_ncycles: int
                Max number of independent flagging cycles, by default 5.
            n_sigma: float, optional
                Flag values greated than n_simga * sigma_hat.
                Where sigma_hat is 1.4826 * MeanAbsoluteDeviation
            n_sigma_rolling: float
                Do a running rms and then flag those regions that have a rms
                higher than n_sigma_rolling*MAD(rmses).
            window_size: int, optional
                Window size for the running rms, by default 11.
            frequencies: List
                List of frequencies.
        """
        if soltype not in self.MODE[mode]:
            raise ValueError(f"Invalid soltype '{soltype}' for mode '{mode}'")

        self.freq = freq
        self.mode = mode
        self.order = order
        self.max_ncycles = max_ncycles
        self.soltype_name = soltype
        self.soltype = self.SOLTYPE[soltype]

        self.flaggers = []
        if n_sigma:
            self.flaggers.append(RMSFlagger(n_sigma))
        if n_sigma_rolling:
            self.flaggers.append(
                RollingRMSFlagger(n_sigma_rolling, window_size)
            )

    def flag_dimension(
        self,
        gains,
        weights,
        antenna=None,
        receptor1=None,
        receptor2=None,
    ):
        """
        Applies flagging to chunk of gaintable with detrending/fitting
        algorithm for the given gain and weight chunk.

        Parameters
        ----------
            gains: xr.DataArray
                Gain solutions.
            weights: xr.DataArray
                Weight of gains.
            antenna: list
                Antenna names
            receptor1: list
                Receptor1 name
            receptor2: list
                Receptor2 name

        Returns
        -------
            weights: xr.DataArray
                Updated weights.
            last_fit_components: dict
                Final fits of soltype.
        """
        weights = weights.copy()
        freq_guess = None
        last_fit_components = None

        for cycle in range(self.max_ncycles):
            if not np.any(weights):
                break

            cycle_flags = np.zeros_like(weights, dtype=bool)
            cycle_sigma = None
            components = {}

            if self.mode == "poly":
                fitter = PhasorPolyFit(self.order, self.freq)
                fit, freq_guess = fitter.fit(gains, weights, freq_guess)

                data_components = self.soltype(gains)
                fit_components = self.soltype(fit)
                components = {
                    key: data_components[key] - fit_components[key]
                    for key in fit_components
                }

                last_fit_components = fit_components

            elif self.mode == "smooth":
                fitter = SmoothingFit(self.order, self.soltype)
                fits = fitter.fit(gains, weights)

                data_components = self.soltype(gains)
                components = {k: data_components[k] - fits[k] for k in fits}
                last_fit_components = fits

            for arr in components.values():
                for flagger in self.flaggers:
                    flags, sigma = flagger.flag(arr, weights)
                    cycle_flags |= flags
                    if sigma is not None and not np.isnan(sigma):
                        cycle_sigma = sigma

            if not np.any(cycle_flags):
                logger.debug("Converged at cycle %d", cycle + 1)
                break

            weights[cycle_flags] = 0

            percent_flagged = (
                100.0 * np.count_nonzero(weights == 0) / weights.size
            )

            logger.info(
                "Gain flagging cycle %d: antenna=%s receptors=[%s,%s] "
                "MAD=%.5f flagged=%.2f%% soltype=%s mode=%s",
                cycle + 1,
                antenna,
                receptor1,
                receptor2,
                cycle_sigma if cycle_sigma is not None else float("nan"),
                percent_flagged,
                self.soltype_name,
                self.mode,
            )

        return weights, last_fit_components


def _flag_wrapper(
    gains,
    weights,
    antenna,
    freq,
    cfg,
    receptor1,
    receptor2,
):
    flagger = GainFlagger(freq=freq, **cfg)
    new_weights, fits = flagger.flag_dimension(
        gains,
        weights,
        antenna=antenna,
        receptor1=receptor1,
        receptor2=receptor2,
    )

    outputs = [new_weights]
    for key in sorted(fits):
        outputs.append(fits[key])

    return tuple(outputs)


def _fit_names(soltype: str):
    return {
        "amplitude": ["amp_fit"],
        "phase": ["phase_fit"],
        "amp-phase": ["amp_fit", "phase_fit"],
        "real-imag": ["real_fit", "imag_fit"],
    }[soltype]


def flag_on_gains(
    gaintable: xr.Dataset,
    soltype: str,
    mode: str,
    order: int,
    max_ncycles: int,
    n_sigma: float,
    n_sigma_rolling: float,
    window_size: int,
    normalize_gains: bool,
    skip_cross_pol: bool,
    apply_flag: bool,
) -> xr.Dataset:
    """
    Solves for gain flagging on gaintable for every receptor combination.
    Optionally applies the weights to the gains.

    Parameters
    ----------
        gaintable: Gaintable
            Gaintable from previous solution.
        soltype: str
            Solution type to flag.
            Can be "real-imag", "phase", "amplitude" or "amp-phase".
        mode: str
            Detrending/fitting algorithm: "smooth", "poly", by default smooth.
        order : int
            Order of the function fitted during detrending.
            If mode=smooth these are the window of the running
            median (0=all axis).
        max_ncycles: int
            Max number of independent flagging cycles, by default 5.
        n_sigma: float
            Flag values greated than n_simga * sigma_hat.
            Where sigma_hat is 1.4826 * MeanAbsoluteDeviation
        n_sigma_rolling: float, optional
            Do a running rms and then flag those regions that have a rms
            higher than n_sigma_rolling*MAD(rmses).
        window_size: int
            Window size for the running rms, by default 11.
        normalize_gains: bool
            Normailize the amplitude and phase before flagging.
        skip_cross_pol: bool
            Cross polarizations is skipped when flagging.
        apply_flag: bool
            Weights are applied to the gains.

    Returns
    -------
        gaintable: Gaintable
            Updated gaintable with weights.
        fits: dict
            All fits generated.
    """

    original_chunks = gaintable.chunks
    gaintable = gaintable.chunk({"frequency": -1})

    freq = gaintable.frequency.data
    fit_names = _fit_names(soltype)

    cfg = dict(
        soltype=soltype,
        mode=mode,
        order=order,
        max_ncycles=max_ncycles,
        n_sigma=n_sigma,
        n_sigma_rolling=n_sigma_rolling,
        window_size=window_size,
    )

    output_core_dims = [["frequency"]] * (1 + len(fit_names))
    output_dtypes = [gaintable.weight.dtype] + [float] * len(fit_names)

    fits = {
        name: xr.zeros_like(gaintable.gain, dtype=float) for name in fit_names
    }

    all_flagged_weights = None

    for receptor1, receptor2 in np.ndindex(
        len(gaintable.receptor1), len(gaintable.receptor2)
    ):
        if skip_cross_pol and receptor1 != receptor2:
            continue

        results = xr.apply_ufunc(
            _flag_wrapper,
            gaintable.gain[0, :, :, receptor1, receptor2],
            gaintable.weight[0, :, :, receptor1, receptor2],
            gaintable.configuration.names.data,
            input_core_dims=[["frequency"], ["frequency"], []],
            output_core_dims=output_core_dims,
            output_dtypes=output_dtypes,
            vectorize=True,
            dask="parallelized",
            kwargs=dict(
                freq=freq,
                cfg=cfg,
                receptor1=gaintable.receptor1[receptor1].data,
                receptor2=gaintable.receptor2[receptor2].data,
            ),
        )

        flagged = results[0]
        all_flagged_weights = (
            flagged
            if all_flagged_weights is None
            else all_flagged_weights * flagged
        )

        for i, name in enumerate(fit_names, start=1):
            fits[name][0, :, :, receptor1, receptor2] = results[i]

    new_weights = gaintable.weight * all_flagged_weights

    if apply_flag:
        new_gain = xr.where(new_weights == 0, 0.0, gaintable.gain)
        gaintable = gaintable.assign(gain=new_gain)

    return (
        gaintable.assign(weight=new_weights).chunk(original_chunks),
        fits,
    )
