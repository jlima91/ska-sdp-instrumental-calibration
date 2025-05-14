import xarray as xr


def sliding_window_smooth(
    gaintable: xr.Dataset, window_size: int, mode: str
) -> xr.Dataset:
    rolled_gain = gaintable.gain.rolling(frequency=window_size, center=True)

    if mode == "mean":
        smooth_gain = rolled_gain.mean()
    else:
        smooth_gain = rolled_gain.median()

    return gaintable.assign(
        {"gain": smooth_gain.chunk(gaintable.gain.chunksizes)}
    )
