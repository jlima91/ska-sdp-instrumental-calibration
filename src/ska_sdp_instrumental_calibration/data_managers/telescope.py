from typing import Annotated

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

    def _create_station(
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
            station_node = self._create_station(name, pos, axes, offsets)
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
        solution_time,
        frequencies,
        station0,
        tile0,
        scale,
    ):
        self._ensure_telescope()

        # Pre-allocate the array
        beams = np.empty(
            (
                self._nstations,
                frequencies.size,
                2,
                2,
            ),
            dtype=np.complex128,
        )

        get_response = self._telescope.station_response

        for stn in range(self._nstations):
            for chan, freq in enumerate(frequencies):
                beams[stn, chan, :, :] = get_response(
                    solution_time,
                    stn,
                    freq,
                    station0,
                    tile0,
                )

        beams *= scale[np.newaxis, :, np.newaxis, np.newaxis]

        return beams

    def single_station_response(
        self,
        solution_time,
        station_idx,
        frequency,
        station0,
        tile0,
    ):
        """
        Calculate the response for a single station at a specific frequency.

        Parameters
        ----------
        solution_time : float
            The time for which to evaluate the response.
        station_idx : int
            Index of the station.
        frequency : float
            The frequency in Hz.
        station0 : array_like
            Direction of the station beam.
        tile0 : array_like
            Direction of the tile beam.

        Returns
        -------
        ndarray
            The 2x2 Jones matrix representing the station response.
        """
        self._ensure_telescope()
        return self._telescope.station_response(
            solution_time,
            station_idx,
            frequency,
            station0,
            tile0,
        )
