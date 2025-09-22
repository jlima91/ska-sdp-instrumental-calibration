import logging
from typing import Literal

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
from dask import delayed
from scipy.ndimage import generic_filter

logger = logging.getLogger()


class FitCurve:
    def __init__(self, mode):
        self.fit = self.__smooth if mode == "" else self.__poly

    def __smooth(self, vals, weights, order, **kwargs):
        vals_smooth = np.copy(vals)
        np.putmask(vals_smooth, weights == 0, np.nan)
        vals_smooth = generic_filter(
            vals_smooth, np.nanmedian, size=order, mode="constant", cval=np.nan
        )
        return vals - vals_smooth

    def __poly(self, vals, weights, order, freq_coord):
        m = np.polyfit(freq_coord, vals, order, w=np.sqrt(weights))
        poly = np.poly1d(m)
        p = poly(freq_coord)
        return vals - p


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
        window_noise: float,
        fix_rms_noise: float,
        frequencies: list[float],
    ):
        self.fit_curve = FitCurve(mode)
        self.sol_type_funcs = self.SOL_TYPE_FUNCS[soltype]
        self.order = order
        self.max_rms = max_rms
        self.fix_rms = fix_rms
        self.max_ncycles = max_ncycles
        self.max_rms_noise = max_rms_noise
        self.window_noise = window_noise
        self.fix_rms_noise = fix_rms_noise
        self.frequencies = frequencies

    def rms_fit(self, vals_detrend, weights):
        rms = 1.4826 * np.nanmedian(np.abs(vals_detrend[(weights != 0)]))
        if np.isnan(rms):
            weights[:] = 0
        elif self.fix_rms > 0:
            flags = abs(vals_detrend) > self.fix_rms
            weights[flags] = 0
        else:
            flags = abs(vals_detrend) > self.max_rms * rms
            weights[flags] = 0
        return weights

    def rm_noise_weights(self, vals_detrend, weights):
        a = np.pad(vals_detrend, self.window_noise / 2, mode="reflect")
        shape = a.shape[:-1] + (
            a.shape[-1] - self.window_noise + 1,
            self.window_noise,
        )
        strides = a.strides + (a.strides[-1],)
        rmses = np.sqrt(
            np.var(
                np.lib.stride_tricks.as_strided(
                    a, shape=shape, strides=strides
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
        return weights

    def flag_dimension(self, gains, weights):
        weights = np.array(weights, copy=True)
        for sol_type_func in self.sol_type_funcs:
            sol_type_data = sol_type_func(gains)
            for i in range(self.max_ncycles):
                # checks on weights
                if (weights == 0).all():
                    # rms = 0.
                    break
                deterend = self.fit_curve.fit(
                    sol_type_data, weights, self.order, self.frequencies
                )
                if self.max_rms > 0 or self.fix_rms > 0:
                    weights = self.rms_fit(deterend, weights)
                if self.max_rms_noise > 0 or self.fix_rms_noise > 0:
                    if self.window_noise % 2 != 1:
                        raise Exception("Window size must be odd.")
                    weights = self.rms_noise_weights(deterend, weights)
        return weights


def _flag(
    vals,
    weights,
    freq_coord,
    order=5,
    mode="poly",
    max_ncycles=3,
    max_rms=3.0,
    max_rms_noise=0.0,
    window_noise=11.0,
    fix_rms=0.0,
    fix_rms_noise=0.0,
):
    weights = np.array(weights, copy=True)
    rms = 0.0
    # for each components of data
    #  for the number iterations
    #
    #    obtain_fit_curve
    #    calculate difference
    #    compute flag based on difference
    # consolidate flags accross components

    for i in range(max_ncycles):

        if (weights == 0).all():
            rms = 0.0
            break

        if mode == "smooth":
            vals_smooth = np.copy(vals)
            np.putmask(vals_smooth, weights == 0, np.nan)
            vals_smooth = generic_filter(
                vals_smooth,
                np.nanmedian,
                size=order,
                mode="constant",
                cval=np.nan,
            )
            vals_detrend = vals - vals_smooth

        elif mode == "poly":
            m = np.polyfit(freq_coord, vals, order, w=np.sqrt(weights))
            poly = np.poly1d(m)
            p = poly(freq_coord)
            vals_detrend = vals - p

        if max_rms > 0 or fix_rms > 0:
            rms = 1.4826 * np.nanmedian(np.abs(vals_detrend[(weights != 0)]))
            if np.isnan(rms):
                weights[:] = 0
            elif fix_rms > 0:
                flags = abs(vals_detrend) > fix_rms
                weights[flags] = 0
            else:
                flags = abs(vals_detrend) > max_rms * rms
                weights[flags] = 0

        if max_rms_noise > 0 or fix_rms_noise > 0:
            rmses = rolling_rms(vals_detrend, window_noise)
            rms = 1.4826 * np.nanmedian(abs(rmses))
            if fix_rms_noise > 0:
                flags = rmses > fix_rms_noise
            else:
                flags = rmses > (max_rms_noise * rms)
            weights[flags] = 0

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
    window_noise: float,
    fix_rms_noise: float,
    apply_the_flag: bool,
) -> xr.Dataset:

    original_chunk = gaintable.chunks
    gaintable = gaintable.chunk({"frequency": -1})

    frequencies = gaintable.gain.coords["frequency"]

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
    updated_weights = None
    for receptor1, receptor2 in np.ndindex(nreceptor1, nreceptor2):
        receptor_weight = xr.apply_ufunc(
            gain_flagger.flag_dimension,
            gaintable.gain[0, :, :, receptor1, receptor2],
            gaintable.weight[0, :, :, receptor1, receptor2],
            input_core_dims=[["frequency"], ["frequency"]],
            output_core_dims=[["frequency"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[gaintable.weight.dtype],
        )

        updated_weights = (
            receptor_weight
            if updated_weights is None
            else updated_weights * receptor_weight
        )
    final_weights = np.repeat(
        updated_weights.data[:, :, np.newaxis], 2, axis=2
    )
    final_weights = np.repeat(final_weights[:, :, :, np.newaxis], 2, axis=3)
    final_weights = final_weights.reshape(-1, *final_weights.shape)

    new_weights = gaintable.weight.copy()
    new_weights.data = final_weights
    new_gain = gaintable["gain"]
    if apply_the_flag:
        new_gain = xr.where(new_weights, 0.0, gaintable["gain"])

    return gaintable.assign(
        {
            "gain": new_gain,
            "weight": new_weights,
        }
    ).chunk(original_chunk)

    new_ds = gaintable.assign(weight=updated_flag)
    apply_the_flag = True
    if apply_the_flag:
        new_gain = xr.where(updated_flag, 0.0, gaintable["gain"])
        new_ds = new_ds.assign(gain=new_gain)

    return new_ds.chunk({"frequency": fchunk})


# def run_flagger(
#     gaintable: xr.Dataset,
#     mode: str = "smooth",
#     order: int = 5,
#     max_rms: float = 1.,
#     max_ncycles: int = 3,
#     soltype: str = "both",
# ) -> xr.Dataset:

#     gaintable = gaintable.chunk({"frequency": -1})

#     flagged = gaintable.map_blocks(
#         flag_on_gains,
#         kwargs=dict(mode="poly", order=order, max_rms=max_rms, max_ncycles=max_ncycles, soltype=soltype),
#         template=gaintable,
#     )

#     return flagged
