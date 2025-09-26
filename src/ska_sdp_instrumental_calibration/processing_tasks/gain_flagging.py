import logging

import numpy as np
import xarray as xr
from scipy.ndimage import generic_filter

logger = logging.getLogger()


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
        "both": [np.absolute, np.angle],
    }

    def __init__(
        self,
        soltype: str,
        mode: str,
        order: int,
        max_rms: float,
        fix_rms: float,
        max_ncycles: int,
        max_rms_noise: float,
        window_noise: int,
        fix_rms_noise: float,
        frequencies: list[float],
    ):
        """
        Generates gain flagger for given soltype and fitting parameters.

        Parameters
        ----------
            soltype: str
                Solution type to flag. Can be "phase", "amplitude"
                or "both".
            mode: str, optional
                Detrending/fitting algorithm: "smooth", "poly".
                By default smooth.
            order : int
                Order of the function fitted during detrending.
                If mode=smooth these are the window of the running
                median (0=all axis).
            max_rms: float, optional
                Rms to clip outliers, by default 5.
            fix_rms: float, optional
                Instead of calculating rms use this value, by default 0.
            max_ncycles: int, optional
                Max number of independent flagging cycles, by default 5.
            max_rms_noise: float, optional
                Do a running rms and then flag those regions that have a rms
                higher than max_rms_noise*rms_of_rmses.
            window_noise: int, optional
                Window size for the running rms, by default 11.
            fix_rms_noise: float, optional
                Instead of calculating rms of the rmses use this value
                (it will not be multiplied by the max_rms_noise).
                By default 0.
            frequencies: List
                List of frequencies.
        """
        self.fit_curve = getattr(FitCurve, mode)

        self.sol_type_funcs = self.SOL_TYPE_FUNCS[soltype]
        self.order = order
        self.max_rms = max_rms
        self.fix_rms = fix_rms
        self.max_ncycles = max_ncycles
        self.max_rms_noise = max_rms_noise
        self.window_noise = window_noise
        self.fix_rms_noise = fix_rms_noise
        self.frequencies = frequencies

    def __rms_flag_weights(self, vals_detrend, weights):
        rms = 1.4826 * np.nanmedian(np.abs(vals_detrend[(weights != 0)]))
        if np.isnan(rms):
            weights[:] = 0
        elif self.fix_rms > 0:
            flags = abs(vals_detrend) > self.fix_rms
            weights[flags] = 0
        else:
            flags = abs(vals_detrend) > self.max_rms * rms
            weights[flags] = 0

        return weights, rms

    def __rms_noise_flag_weights(self, vals_detrend, weights):
        if self.window_noise % 2 != 1:
            raise Exception("Window size must be odd.")

        detrend_pad = np.pad(
            vals_detrend, self.window_noise // 2, mode="reflect"
        )
        shape = detrend_pad.shape[:-1] + (
            detrend_pad.shape[-1] - self.window_noise + 1,
            self.window_noise,
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

        rms = 1.4826 * np.nanmedian(abs(rmses))

        if self.fix_rms_noise > 0:
            flags = rmses > self.fix_rms_noise
        else:
            flags = rmses > (self.max_rms_noise * rms)

        weights[flags] = 0
        return weights, rms

    def flag_dimension(self, gains, weights, antenna, receptor1, receptor2):
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
        """

        soltype = {"absolute": "amplitude", "angle": "phase"}
        weights = np.array(weights, copy=True)
        for sol_type_func in self.sol_type_funcs:
            rms = 0.0
            sol_type_data = sol_type_func(gains)
            for _ in range(self.max_ncycles):
                if all(weights == 0):
                    break

                deterend = sol_type_data - self.fit_curve(
                    sol_type_data, weights, self.order, self.frequencies
                )

                if self.max_rms > 0 or self.fix_rms > 0:
                    weights, rms = self.__rms_flag_weights(deterend, weights)

                if self.max_rms_noise > 0 or self.fix_rms_noise > 0:
                    weights, rms = self.__rms_noise_flag_weights(
                        deterend, weights
                    )

            logger.info(
                f"Gain flagging: Antenna {antenna} "
                f"receptors [{receptor1},{receptor2}]- "
                f"rms: {rms:.5f}, "
                f"for {soltype[sol_type_func.__name__]}."
            )
        return weights


def flag_on_gains(
    gaintable: xr.Dataset,
    soltype: str,
    mode: str,
    order: int,
    max_rms: float,
    fix_rms: float,
    max_ncycles: int,
    max_rms_noise: float,
    window_noise: int,
    fix_rms_noise: float,
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
            Solution type to flag. Can be "phase", "amplitude" or "both".
        mode: str, optional
            Detrending/fitting algorithm: "smooth", "poly", by default smooth.
        order : int
            Order of the function fitted during detrending.
            If mode=smooth these are the window of the running
            median (0=all axis).
        skip_cross_pol: bool
            Cross polarizations is skipped when flagging.
        max_rms: float, optional
            Rms to clip outliers, by default 5.
        fix_rms: float, optional
            Instead of calculating rms use this value, by default 0.
        max_ncycles: int, optional
            Max number of independent flagging cycles, by default 5.
        max_rms_noise: float, optional
            Do a running rms and then flag those regions that have a rms
            higher than max_rms_noise*rms_of_rmses.
        window_noise: int, optional
            Window size for the running rms, by default 11.
        fix_rms_noise: float, optional
            Instead of calculating rms of the rmses use this value
            (it will not be multiplied by the max_rms_noise), by default 0.
        apply_flag: bool
            Weights are applied to the gains.

    Returns
    -------
        gaintable: Gaintable
            Updated gaintable with weights.
    """

    original_chunk = gaintable.chunks
    gaintable = gaintable.chunk({"frequency": -1})

    frequencies = gaintable.gain.coords["frequency"]
    logger.info(f"Gain flagging for mode: {soltype} started")
    gain_flagger = GainFlagger(
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
    nreceptor1 = len(gaintable.receptor1)
    nreceptor2 = len(gaintable.receptor2)
    all_flagged_weights = None

    for receptor1, receptor2 in np.ndindex(nreceptor1, nreceptor2):
        if receptor1 != receptor2 and skip_cross_pol:
            continue

        receptor_weight = xr.apply_ufunc(
            gain_flagger.flag_dimension,
            gaintable.gain[0, :, :, receptor1, receptor2],
            gaintable.weight[0, :, :, receptor1, receptor2],
            gaintable.configuration.names.data,
            input_core_dims=[["frequency"], ["frequency"], []],
            output_core_dims=[["frequency"]],
            vectorize=True,
            dask="parallelized",
            kwargs=dict(
                receptor1=gaintable.receptor1[receptor1].data,
                receptor2=gaintable.receptor2[receptor2].data,
            ),
            output_dtypes=[gaintable.weight.dtype],
        )

        all_flagged_weights = (
            receptor_weight
            if all_flagged_weights is None
            else all_flagged_weights * receptor_weight
        )

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
    new_weights.data = flagged_weights_data

    if apply_flag:
        new_gain = xr.where(new_weights == 0, 0.0, gaintable["gain"])
        return gaintable.assign(
            {
                "gain": new_gain,
                "weight": new_weights,
            }
        ).chunk(original_chunk)

    return gaintable.assign(
        {
            "weight": new_weights,
        }
    ).chunk(original_chunk)
