"""
MSv4 (xradio) schema for calibration solutions (GainCalibrationSetXds).
"""

from typing import Literal, Union, cast

import numpy
import xarray
from numpy.typing import NDArray
from xradio.measurement_set.schema import (
    AntennaNameArray,
    Polarization,
    PolarizationArray,
)
from xradio.schema.bases import xarray_dataarray_schema, xarray_dataset_schema
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof

Time = Literal["time"]
AntennaName = Literal["antenna_name"]
Frequency = Literal["frequency"]


@xarray_dataarray_schema
class GainCalibrationParameterArray:
    """
    Scalar gains or flattened 2x2 Jones matrices for Gains
    """

    data: Data[
        tuple[
            Time,
            AntennaName,
            Frequency,
            Polarization,
        ],
        Union[
            numpy.complex64,
            numpy.complex128,
        ],
    ]

    time: Coord[Time, float]
    antenna_name: Coordof[AntennaNameArray]
    frequency: Coord[Frequency, float]
    polarization: Coordof[PolarizationArray]


@xarray_dataarray_schema
class GainCalibrationWeightArray:
    """Gain weights, same shape as gain"""

    data: Data[
        tuple[
            Time,
            AntennaName,
            Frequency,
            Polarization,
        ],
        Union[
            numpy.float32,
            numpy.float64,
        ],
    ]

    time: Coord[Time, float]
    antenna_name: Coordof[AntennaNameArray]
    frequency: Coord[Frequency, float]
    polarization: Coordof[PolarizationArray]


@xarray_dataarray_schema
class GainCalibrationResidualArray:
    """Fit residuals."""

    data: Data[
        tuple[
            Time,
            Frequency,
            Polarization,
        ],
        Union[
            numpy.float32,
            numpy.float64,
        ],
    ]

    time: Coord[Time, float]
    frequency: Coord[Frequency, float]
    polarization: Coordof[PolarizationArray]


@xarray_dataarray_schema
class GainCalibrationIntervalArray:
    """Solution interval slices"""

    data: Data[
        tuple[Time],
        Union[
            numpy.float32,
            numpy.float64,
        ],
    ]

    time: Coord[Time, float]


@xarray_dataset_schema
class GainCalibrationSetXds:
    """
    MSv4 container for calibration solutions. class:`xarray.Dataset`.
    """

    CALPARAM_GAIN: Dataof[GainCalibrationParameterArray]
    CALPARAM_WEIGHT: Dataof[GainCalibrationWeightArray]
    CALPARAM_RESIDUAL: Dataof[GainCalibrationResidualArray]
    CALPARAM_INTERVAL: Dataof[GainCalibrationIntervalArray]

    time: Coord[Time, float]
    antenna_name: Coordof[AntennaNameArray]
    frequency: Coord[Frequency, float]
    polarization: Coordof[PolarizationArray]

    jones_type: Attr[Literal["T", "G", "B"]]
    type: Attr[Literal["gain_table"]] = "gain_table"

    @classmethod
    def constructor(
        cls,
        *,
        gain: NDArray,
        weight: NDArray,
        residual: NDArray,
        time: NDArray,
        interval: NDArray,
        frequency: NDArray,
        antenna_name: NDArray[numpy.str_],
        polarization: NDArray[numpy.str_],
        jones_type: Literal["T", "G", "B"],
    ) -> xarray.Dataset:
        """
        Create a schema-checked GainCalibrationSetXds dataset.

        Parameters
        ----------
        gain
            Complex gains ``[ntimes, nants, nchan, npol]``.
        weight
            Weights of gains, same shape as ``gain``.
        residual
            Residuals of fit ``[ntimes, nchan, npol]``.
        time
            Centroids of solutions, in seconds elapsed since the MJD
            reference epoch ``[ntimes]``.
        interval
            Intervals of validity in seconds ``[ntimes]``.
        frequency
            Channel frequencies in Hz ``[nchan]``.
        antenna_name
            Antenna names ``[nants]``.
        polarization
            Polarization labels ``[npol]``, e.g. ``XX XY YX YY``. The
            values are not checked; the caller must pass valid
            PolarizationArray names.
        jones_type
            Capital letter denoting the Jones term this table represents.

        Returns
        -------
            Schema-checked xarray.Dataset.

        Raises
        ------
        ValueError
            If the number of antenna names or polarization labels does
            not match the shape of ``gain``.
        """
        antenna_name = numpy.asarray(antenna_name, dtype=str)
        polarization = numpy.asarray(polarization, dtype=str)

        if len(antenna_name) != gain.shape[1]:
            raise ValueError(
                f"Got {len(antenna_name)} antenna names for "
                f"{gain.shape[1]} antennas in gain"
            )

        if len(polarization) != gain.shape[-1]:
            raise ValueError(
                f"Got {len(polarization)} polarization labels for "
                f"{gain.shape[-1]} polarizations in gain"
            )

        fields = dict(
            CALPARAM_GAIN=gain,
            CALPARAM_WEIGHT=weight,
            CALPARAM_RESIDUAL=residual,
            CALPARAM_INTERVAL=interval,
            time=numpy.asarray(time),
            antenna_name=antenna_name,
            frequency=numpy.asarray(frequency),
            polarization=polarization,
            jones_type=jones_type,
        )
        # The xradio decorator makes calling the class build and schema-check
        # an xarray.Dataset (not an instance of this class)
        return cast(xarray.Dataset, cls(**fields))
