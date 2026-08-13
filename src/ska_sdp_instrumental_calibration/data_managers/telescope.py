from typing import Annotated, Optional

import everybeam as eb
import numpy as np
import numpy.typing as npt

from .visibility import read_antenna_response_data


class Telescope:
    """Telescope model representation."""

    def __init__(
        self,
        ms_path: str,
        element_response_model,
    ):
        """
        Initialize the Telescope model.

        Parameters
        ----------
        ms_path : str
            Path to the measurement set.
        element_response_model : str
            Name of the element response model.
        """
        self._telescope = None
        self._element_response_model = getattr(
            eb.ElementResponseModel, element_response_model
        )

        self.response_data = read_antenna_response_data(ms_path)
        self._nstations = 0

    def _create_station_node(
        self,
        name: str,
        position: Annotated[npt.NDArray[np.float64], "shape (3,)"],
        coordinate_axes: Annotated[npt.NDArray[np.float64], "shape (3, 3)"],
        element_offsets: Annotated[npt.NDArray[np.float64], "shape (N, 3)"],
    ):
        """
        Create a single station node with given elements and coordinate system.

        Parameters
        ----------
        name : str
            Name of the station.
        position : array_like, shape (3,)
            The (x, y, z) position of the station.
        coordinate_axes : array_like, shape (3, 3)
            The coordinate axes of the station.
        element_offsets : array_like, shape (N, 3)
            The (x, y, z) offsets of N elements within the station.

        Returns
        -------
        eb.StationNode
            The constructed station node object.
        """
        station_node = eb.StationNode(
            coordinate_system=eb.StationCoordinateSystem(
                position, coordinate_axes
            ),
            name=f"{name}",
        )
        add_child_element = station_node.add_child_element
        for offset in element_offsets:
            add_child_element(offset)

        return station_node

    def _initialize_everybeam_telescope(self):
        """
        Create an OSKAR telescope model containing multiple stations.
        """
        station_names = self.response_data["station_names"]
        positions = self.response_data["positions"]
        coordinate_axes = self.response_data["coordinate_axes"]
        element_offsets = self.response_data["element_offsets"]
        delay_directions = self.response_data["delay_directions"]

        self._nstations = station_names.size
        root_node = eb.StationNode()
        add_child_node = root_node.add_child_node
        for name, pos, axes, offsets in zip(
            station_names, positions, coordinate_axes, element_offsets
        ):
            station_node = self._create_station_node(name, pos, axes, offsets)
            add_child_node(station_node, pos)
        options = eb.Options()
        options.element_response_model = self._element_response_model

        self._telescope = eb.create_telescope(
            eb.TelescopeType.OSKAR,
            options,
            root_node,
            delay_directions=delay_directions,
        )

    def _ensure_telescope(self):
        """
        Ensure that the internal telescope model is instantiated.
        """
        if self._telescope is None:
            self._initialize_everybeam_telescope()

    @property
    def type(self) -> type:
        """
        Get the underlying telescope model type.

        Returns
        -------
        type
            Type of the created telescope object.
        """
        self._ensure_telescope()
        return type(self._telescope)

    def station_response(
        self,
        solution_time: np.float64 | float,
        frequencies: npt.NDArray | float | np.float64,
        station0: Annotated[npt.NDArray[np.float64], "shape (3,)"],
        tile0: Annotated[npt.NDArray, "shape (3,)"],
        scale: Optional[npt.NDArray] = None,
        station_idx: Optional[int] = None,
    ):
        """
        Calculate the station response beams for all stations and frequencies.

        Parameters
        ----------
        solution_time : float or np.float64
            Timestamp for the solution.
        frequencies : npt.NDArray
            1D array of frequencies.
        station0 : npt.NDArray[np.float64]
            Direction vector for station beam pointing, shape (3,).
        tile0 : npt.NDArray[np.float64]
            Direction vector for tile beam pointing, shape (3,).
        scale : npt.NDArray, optional
            Scaling array per frequency to apply to beams, default is None.

        Returns
        -------
        npt.NDArray[np.complex128]
            Complex beam matrices with shape
            (nstations, nfrequencies, 2, 2).
        """

        self._ensure_telescope()
        frequencies = np.atleast_1d(frequencies)
        solution_times = np.atleast_1d(solution_time)

        stations = (
            range(self._nstations) if station_idx is None else [station_idx]
        )

        beams = self._telescope.station_response(
            solution_times,
            stations,
            frequencies,
            station0,
            tile0,
        ).squeeze(axis=0)

        if scale is not None:
            beams *= scale[np.newaxis, :, np.newaxis, np.newaxis]

        return beams
