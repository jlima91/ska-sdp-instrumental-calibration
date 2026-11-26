"""
Observation parameters of a field, computed from the timestamps of its
measurement sets. The timestamps, channels, stations and measurement sets
are looked up by :mod:`..repository.metadata`.
"""

import numpy as np

from ..domain.metadata import ObservationParameters


def get_observation_parameters(
    timestamps: list[list[float]], channels: int | None
) -> ObservationParameters:
    """
    Get the observation parameters of a field from the timestamps of its
    measurement sets.

    The integration time is the median spacing of the timestamps of the
    first measurement set. The observation spans the timestamps of all
    measurement sets, which mark the centre of each integration.

    Parameters
    ----------
    timestamps : list[list[float]]
        Timestamps of each measurement set of the field, at least one, in
        MJD seconds, in any order. See
        :func:`..repository.metadata.get_timestamps`.
    channels : int or None
        Number of frequency channels, or None if not recorded.

    Returns
    -------
    ObservationParameters
        Start time, observation length, integration time and channels of
        the field.
    """
    # Measurement sets of a field share the integration time, so it is
    # taken from the first one
    integration_time = float(np.median(np.diff(timestamps[0])))

    # Timestamps mark the centre of each integration, so the observation
    # spans half an integration beyond the first and last ones
    start_time = (
        min(min(ms_timestamps) for ms_timestamps in timestamps)
        - integration_time / 2
    )
    end_time = (
        max(max(ms_timestamps) for ms_timestamps in timestamps)
        + integration_time / 2
    )

    return ObservationParameters(
        start_time=float(start_time),
        observation_length=float(end_time - start_time),
        integration_time=integration_time,
        channels=channels,
    )
