import logging

import xarray as xr

logger = logging.getLogger()


def sliding_window_smooth(
    gaintable: xr.Dataset, window_size: int, mode: str
) -> xr.Dataset:
    rolled_gain = gaintable.CALPARAM_GAIN.rolling(
        frequency=window_size, center=True
    )

    if mode == "mean":
        logger.info("Using sliding window smooth with mean mode.")
        smooth_gain = rolled_gain.mean()
    elif mode == "median":
        logger.info("Using sliding window smooth with median mode.")
        smooth_gain = rolled_gain.median()
    else:
        raise ValueError(f"Unsupported sliding window smooth mode {mode}")

    return gaintable.assign(
        {
            "CALPARAM_GAIN": smooth_gain.chunk(
                gaintable.CALPARAM_GAIN.chunksizes
            )
        }
    )
