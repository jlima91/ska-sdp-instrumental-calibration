"""
MSv4 (xradio) schema for calibration solutions (GainCalibrationSetXds).
"""

from typing import Literal, Optional, Sequence, Union, cast

import numpy
import xarray
from numpy.typing import NDArray
from ska_sdp_datamodels.configuration import Configuration
from ska_sdp_datamodels.science_data_model import (
    QualityAssessment,
    ReceptorFrame,
)
from ska_sdp_datamodels.xarray_accessor import XarrayAccessorMixin
from xradio.measurement_set.schema import (
    AntennaNameArray,
    Polarization,
    PolarizationArray,
)
from xradio.schema.bases import xarray_dataarray_schema, xarray_dataset_schema
from xradio.schema.typing import Attr, Coord, Coordof, Data, Dataof

from ...numpy_processors._utils import NPOL_TO_NREC

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

    jones_type: Attr[Literal["T", "G", "B", "K"]]
    type: Attr[Literal["gain_table"]] = "gain_table"

    @classmethod
    def constructor(
        cls,
        gain: Optional[NDArray] = None,
        time: Optional[NDArray] = None,
        interval: Optional[NDArray] = None,
        weight: Optional[NDArray] = None,
        residual: Optional[NDArray] = None,
        frequency: Optional[NDArray] = None,
        receptor_frame: Optional[
            Union[ReceptorFrame, Sequence[ReceptorFrame]]
        ] = None,
        configuration: Optional[Configuration] = None,
        jones_type: Literal["T", "G", "B", "K"] = "T",
    ) -> xarray.Dataset:
        """
        Create a GainCalibrationSetXds dataset.

        :param gain: Complex gains [ntimes, nants, nchan, npol]
        :param time: Centroids of solutions, in seconds elapsed since the MJD
            reference epoch [ntimes]
        :param interval: Intervals of validity in seconds [ntimes]
        :param weight: Weights of gains [ntimes, nants, nchan, npol]
        :param residual: Residuals of fit [ntimes, nchan, npol]
        :param frequency: Channel frequencies in Hz [nchan]
        :param receptor_frame: Measured and ideal (model) data receptor
            frames. If None, use a linear receptor frame for both. If
            ReceptorFrame instance, use it for both. If two-element sequence,
            interpret as [receptor1, receptor2].
        :param configuration: Array configuration, used for antenna names.
            If None, antenna names are the antenna indices as strings.
        :param jones_type: Capital letter denoting the Jones term this
            GainTable will represent.
        :return: Schema-checked xarray.Dataset
        """
        if gain is None:
            raise ValueError("gain is required")

        nants = gain.shape[1]
        if configuration is not None:
            antenna_names = configuration.names.data
        else:
            antenna_names = [str(ant) for ant in range(nants)]

        if receptor_frame is None:
            receptor_frame = ReceptorFrame("linear")

        if isinstance(receptor_frame, ReceptorFrame):
            receptor1, receptor2 = (receptor_frame, receptor_frame)
        else:
            receptor1, receptor2 = receptor_frame
            if not receptor1.nrec == receptor2.nrec:
                raise ValueError(
                    "When providing two receptor frames, "
                    "they must have the same number of polarisation hands"
                )

        fields = dict(
            CALPARAM_GAIN=gain,
            CALPARAM_WEIGHT=weight,
            CALPARAM_RESIDUAL=residual,
            CALPARAM_INTERVAL=interval,
            time=numpy.asarray(time),
            antenna_name=numpy.asarray(antenna_names, dtype=str),
            frequency=numpy.asarray(frequency),
            polarization=numpy.asarray(
                [
                    f"{r1}{r2}"
                    for r1 in receptor1.names
                    for r2 in receptor2.names
                ],
                dtype=str,
            ),
            jones_type=jones_type,
        )
        # The xradio decorator makes calling the class build and schema-check
        # an xarray.Dataset (not an instance of this class)
        return cast(xarray.Dataset, cls(**fields))


@xarray.register_dataset_accessor("calibration_set")
class GainCalibrationSetAccessor(XarrayAccessorMixin):
    """GainCalibrationSetXds property accessor"""

    @property
    def ntimes(self) -> int:
        """Number of times (i.e. rows) in this table"""
        return self._obj.sizes["time"]

    @property
    def nants(self) -> int:
        """Number of dishes/stations"""
        return self._obj.sizes["antenna_name"]

    @property
    def nchan(self) -> int:
        """Number of channels"""
        return self._obj.sizes["frequency"]

    @property
    def nrec(self) -> int:
        """Number of polarisation in receptors"""
        return NPOL_TO_NREC[self._obj.sizes["polarization"]]

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy GainCalibrationSetXds

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set gain data to zero in copied object
        """
        new_gt = self._obj.copy(deep=deep, data=data)
        if zero:
            new_gt["CALPARAM_GAIN"].data[...] = 0.0
        return new_gt

    def qa_gain_table(self, context=None) -> QualityAssessment:
        """Assess the quality of a gaintable

        :return: QualityAssessment
        """
        weight_data = self._obj.WEIGHT.data
        if numpy.max(weight_data) <= 0.0:
            raise ValueError("qa_gain_table: All gaintable weights are zero")

        gain_data = self._obj.CALPARAM_GAIN.data
        agt = numpy.abs(gain_data[weight_data > 0.0])
        pgt = numpy.angle(gain_data[weight_data > 0.0])
        rgt = self._obj.RESIDUAL.data[numpy.sum(weight_data, axis=1) > 0.0]
        data = {
            "shape": self._obj.CALPARAM_GAIN.shape,
            "maxabs-amp": numpy.max(agt),
            "minabs-amp": numpy.min(agt),
            "rms-amp": numpy.std(agt),
            "medianabs-amp": numpy.median(agt),
            "maxabs-phase": numpy.max(pgt),
            "minabs-phase": numpy.min(pgt),
            "rms-phase": numpy.std(pgt),
            "medianabs-phase": numpy.median(pgt),
            "residual": numpy.max(rgt),
        }
        qa = QualityAssessment(
            origin="qa_gain_table", data=data, context=context
        )
        return qa
