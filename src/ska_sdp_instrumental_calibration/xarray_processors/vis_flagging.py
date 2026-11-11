import numpy as np
import xarray as xr


# TODO: Get the logic checked by domain expert
def flagged_time_count(vis: xr.Dataset) -> xr.DataArray:
    """
    Number of flagged time steps per baseline and channel.

    A sample is flagged where its solver weight is 0, as in
    AlternativeSolver.solve: weight * (1 - flags), and the whole sample is
    flagged if any of its polarisations is.

    Parameters
    ----------
    vis: Visibility
        Visibility dataset.

    Returns
    -------
    xr.DataArray
        Lazy flagged time step count, with dimensions
        (``baselineid``, ``frequency``).
    """
    # Same as weight * (1 - flags) == 0, but as booleans, so the dask
    # graph holds 1 byte per sample instead of a float copy of the weights
    is_flagged = ((vis.weight == 0) | vis.flags).any(dim="polarisation")

    return is_flagged.sum(dim="time")


def _cross_correlations(
    flagged_count: np.ndarray, antenna1: np.ndarray, antenna2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keep only the cross-correlation baselines.

    Only cross-correlations are used by the calibration solvers
    (gen_coherency_products). Autocorrelations are also padded with
    weight 0 by load_data when missing from the MS.
    """
    is_cross = antenna1 != antenna2
    return flagged_count[is_cross], antenna1[is_cross], antenna2[is_cross]


def flagged_sample_counts(
    flagged_count: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ntimes: int,
    nstations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Number of flagged and of all visibility samples per station, over the
    cross-correlations of one visibility.

    Parameters
    ----------
    flagged_count: np.ndarray
        Flagged time step count per baseline and channel.
    antenna1: np.ndarray
        First station index of each baseline.
    antenna2: np.ndarray
        Second station index of each baseline.
    ntimes: int
        Number of time steps.
    nstations: int
        Number of stations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flagged and total sample count per station.
    """
    flagged_count, antenna1, antenna2 = _cross_correlations(
        flagged_count, antenna1, antenna2
    )
    flagged_per_baseline = flagged_count.sum(axis=1)
    samples_per_baseline = ntimes * flagged_count.shape[1]

    # np.bincount(idx, weights=w)[k] sums w over all baselines where
    # idx == k. Each baseline counts for both of its stations, hence
    # antenna1 + antenna2.
    flagged = np.bincount(
        antenna1, weights=flagged_per_baseline, minlength=nstations
    ) + np.bincount(
        antenna2, weights=flagged_per_baseline, minlength=nstations
    )
    total = (
        np.bincount(antenna1, minlength=nstations)
        + np.bincount(antenna2, minlength=nstations)
    ) * samples_per_baseline

    return flagged, total


def unsolvable_gain_counts(
    flagged_count: np.ndarray,
    antenna1: np.ndarray,
    antenna2: np.ndarray,
    ntimes: int,
    nstations: int,
) -> tuple[np.ndarray, int]:
    """
    Number of gain solutions per station that cannot be solved, because
    all visibilities of the station in that channel are flagged, and of
    all gain solutions, for one visibility.

    This counts in the same unit as the flagged gains of a gaintable: one
    solution per station and channel, as bandpass solves one solution over
    all time steps of a visibility.

    Parameters
    ----------
    flagged_count: np.ndarray
        Flagged time step count per baseline and channel.
    antenna1: np.ndarray
        First station index of each baseline.
    antenna2: np.ndarray
        Second station index of each baseline.
    ntimes: int
        Number of time steps.
    nstations: int
        Number of stations.

    Returns
    -------
    tuple[np.ndarray, int]
        Unsolvable gain count per station, and the number of gain
        solutions of a station.
    """
    flagged_count, antenna1, antenna2 = _cross_correlations(
        flagged_count, antenna1, antenna2
    )
    nchannels = flagged_count.shape[1]
    # A baseline channel has data when not all its time steps are flagged
    has_data = (flagged_count < ntimes).astype(int)

    # Number of baselines with data per station and channel, in one pass
    # over the baselines: np.add.at adds each baseline row to both of its
    # stations, like np.bincount for the sample counts
    baselines_with_data = np.zeros((nstations, nchannels), dtype=int)
    np.add.at(baselines_with_data, antenna1, has_data)
    np.add.at(baselines_with_data, antenna2, has_data)

    # A channel is unsolvable when no baseline of the station has data
    unsolvable = (baselines_with_data == 0).sum(axis=1)

    return unsolvable, nchannels
