import logging

import dask
import numpy as np
import xarray as xr
from scipy.ndimage import generic_filter

from ._utils import normalize_data

logger = logging.getLogger()


@dask.delayed
def log_flaging_statistics(weights, initial_weights, antenna_name):
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


class FitCurve:
    @staticmethod
    def smooth(vals, weights, order, _):
        """
        Applies smoothing filter to gains.
        Parameters
        ----------
            vals: Array
                Gains
            weights: Array
                Weights of gains
            order: int
                Order of the function.
        Returns
        -------
            Fit of gains after applying smooth.
        """
        vals_smooth = np.copy(vals)
        np.putmask(vals_smooth, weights == 0, np.nan)
        return generic_filter(
            vals_smooth, np.nanmedian, size=order, mode="constant", cval=np.nan
        )

    @staticmethod
    def poly(vals, weights, order, freq_coord):
        """
        Fits polynominal to gains.

        Parameters
        ----------
            vals: Array
                Gains
            weights: Array
                Weights of gains
            order: int
                Order of the function.
        Returns
        -------
            Fit of gains after applying poly fit.
        """
        coeff = np.polyfit(freq_coord, vals, order, w=np.sqrt(weights))
        poly = np.poly1d(coeff)
        return poly(freq_coord)


class GainFlagger:
    SOL_TYPE_FUNCS = {
        "amplitude": [np.absolute],
        "phase": [np.angle],
        "amp-phase": [np.absolute, np.angle],
        "real-imag": [np.real, np.imag],
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
        frequencies: list[float],
        normalize_gains: bool,
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
            normalize_gains: bool
                Normailize the amplitude and phase before flagging.
        """
        self.fit_curve = getattr(FitCurve, mode)

        self.sol_type_funcs = self.SOL_TYPE_FUNCS[soltype]
        self.order = order
        self.n_sigma = n_sigma
        self.n_sigma_rolling = n_sigma_rolling
        self.max_ncycles = max_ncycles
        self.window_size = window_size
        self.frequencies = frequencies
        self.normalize_gains = normalize_gains

    def __rms_flag_weights(self, vals_detrend, weights):
        sigma_hat = 1.4826 * np.nanmedian(np.abs(vals_detrend[(weights != 0)]))

        if np.isnan(sigma_hat):
            flags = np.ones(weights.shape) == 1

        flags = abs(vals_detrend) > self.n_sigma * sigma_hat

        return flags, sigma_hat

    def __rms_noise_flag_weights(self, vals_detrend, weights):
        if self.window_size % 2 != 1:
            raise Exception("Window size must be odd.")

        detrend_pad = np.pad(
            vals_detrend, self.window_size // 2, mode="reflect"
        )
        shape = detrend_pad.shape[:-1] + (
            detrend_pad.shape[-1] - self.window_size + 1,
            self.window_size,
        )
        strides = (
            detrend_pad.strides
            + (  # pylint: disable=unsubscriptable-object
                detrend_pad.strides[
                    -1
                ],  # pylint: disable=unsubscriptable-object
            )
        )
        rmses = np.sqrt(
            np.var(
                np.lib.stride_tricks.as_strided(
                    detrend_pad, shape=shape, strides=strides
                ),
                -1,
            )
        )

        sigma_hat = 1.4826 * np.nanmedian(abs(rmses))

        flags = rmses > (self.n_sigma_rolling * sigma_hat)

        return flags, sigma_hat

    def flag_dimension(self, weights, gains, antenna, receptor1, receptor2):
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
            amp_fits: xr.DataArray
                Amplitude fit
            phase_fits: xr.DataArray
                Phase fit
            real_fits: xr.DataArray
                Real fit
            imag_fits: xr.DataArray
                Imaginary fit
        """

        soltype = {
            "absolute": "amplitude",
            "angle": "phase",
            "real": "real",
            "imag": "imag",
        }

        weights = np.array(weights, copy=True)
        flagged_weights = np.ones(weights.shape)
        amp_fit = np.zeros_like(gains, dtype=float)
        phase_fit = np.zeros_like(gains, dtype=float)
        real_fit = np.zeros_like(gains, dtype=float)
        imag_fit = np.zeros_like(gains, dtype=float)

        for sol_type_func in self.sol_type_funcs:
            rms = 0.0
            sol_type_data = sol_type_func(gains)

            if self.normalize_gains and sol_type_func == np.absolute:
                sol_type_data = normalize_data(sol_type_data)

            for _ in range(self.max_ncycles):
                if all(weights == 0):
                    break
                curve_fit = self.fit_curve(
                    sol_type_data, weights, self.order, self.frequencies
                )

                if sol_type_func == np.absolute:
                    amp_fit = curve_fit
                elif sol_type_func == np.angle:
                    phase_fit = curve_fit
                elif sol_type_func == np.real:
                    real_fit = curve_fit
                elif sol_type_func == np.imag:
                    imag_fit = curve_fit

                deterend = sol_type_data - self.fit_curve(
                    sol_type_data, weights, self.order, self.frequencies
                )

                if self.n_sigma is not None and self.n_sigma > 0:
                    flags, rms = self.__rms_flag_weights(deterend, weights)
                    weights[flags] = 0
                    flagged_weights[flags] = 0

                if (
                    self.n_sigma_rolling is not None
                    and self.n_sigma_rolling > 0
                ):
                    flags, rms = self.__rms_noise_flag_weights(
                        deterend, weights
                    )
                    weights[flags] = 0
                    flagged_weights[flags] = 0

            percent_flagged = (
                ((flagged_weights == 0).sum()) / (flagged_weights.size) * 100
            )

            logger.info(
                f"Gain flagging: Antenna {antenna} "
                f"receptors [{receptor1},{receptor2}] with "
                f"MAD of {rms:.5f} and {percent_flagged:.2f}% gains flagged "
                f"for {soltype[sol_type_func.__name__]}."
            )
        return flagged_weights, amp_fit, phase_fit, real_fit, imag_fit


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

    original_chunk = gaintable.chunks
    gaintable = gaintable.chunk({"frequency": -1})

    frequencies = gaintable.gain.coords["frequency"]
    logger.info(f"Gain flagging for mode: {soltype} started")
    gain_flagger = GainFlagger(
        soltype,
        mode,
        order,
        max_ncycles,
        n_sigma,
        n_sigma_rolling,
        window_size,
        frequencies,
        normalize_gains,
    )
    nreceptor1 = len(gaintable.receptor1)
    nreceptor2 = len(gaintable.receptor2)
    all_flagged_weights = None

    fits = dict()
    fits["amp_fit"] = xr.full_like(gaintable.gain, fill_value=0, dtype=float)
    fits["phase_fit"] = xr.full_like(gaintable.gain, fill_value=0, dtype=float)
    fits["real_fit"] = xr.full_like(gaintable.gain, fill_value=0, dtype=float)
    fits["imag_fit"] = xr.full_like(gaintable.gain, fill_value=0, dtype=float)

    for receptor1, receptor2 in np.ndindex(nreceptor1, nreceptor2):
        if receptor1 != receptor2 and skip_cross_pol:
            continue

        receptor_weight_flag, amp_fit, phase_fit, real_fit, imag_fit = (
            xr.apply_ufunc(
                gain_flagger.flag_dimension,
                gaintable.weight[0, :, :, receptor1, receptor2],
                gaintable.gain[0, :, :, receptor1, receptor2],
                gaintable.configuration.names.data,
                input_core_dims=[["frequency"], ["frequency"], []],
                output_core_dims=[
                    ["frequency"],
                    ["frequency"],
                    ["frequency"],
                    ["frequency"],
                    ["frequency"],
                ],
                vectorize=True,
                dask="parallelized",
                kwargs=dict(
                    receptor1=gaintable.receptor1[receptor1].data,
                    receptor2=gaintable.receptor2[receptor2].data,
                ),
                output_dtypes=[
                    gaintable.weight.dtype,
                    float,
                    float,
                    float,
                    float,
                ],
            )
        )

        all_flagged_weights = (
            receptor_weight_flag
            if all_flagged_weights is None
            else all_flagged_weights * receptor_weight_flag
        )

        fits["amp_fit"][0, :, :, receptor1, receptor2] = amp_fit
        fits["phase_fit"][0, :, :, receptor1, receptor2] = phase_fit
        fits["real_fit"][0, :, :, receptor1, receptor2] = real_fit
        fits["imag_fit"][0, :, :, receptor1, receptor2] = imag_fit

    flagged_weights_data = np.repeat(
        all_flagged_weights.data[:, :, np.newaxis], nreceptor1, axis=2
    )
    flagged_weights_data = np.repeat(
        flagged_weights_data[:, :, :, np.newaxis], nreceptor2, axis=3
    )
    flagged_weights_data = flagged_weights_data.reshape(
        -1, *flagged_weights_data.shape
    )

    new_weights = gaintable.weight.copy()
    new_weights.data = gaintable.weight.data * flagged_weights_data

    if apply_flag:
        new_gain = xr.where(new_weights == 0, 0.0, gaintable["gain"])
        return (
            gaintable.assign(
                {
                    "gain": new_gain,
                    "weight": new_weights,
                }
            ).chunk(original_chunk),
            fits,
        )

    return (
        gaintable.assign(
            {
                "weight": new_weights,
            }
        ).chunk(original_chunk),
        fits,
    )
