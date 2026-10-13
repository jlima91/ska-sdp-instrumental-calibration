# pylint:disable=c-extension-no-member
import threading
from typing import Annotated, Optional

import everybeam as eb
import numpy as np
import numpy.typing as npt

from .visibility import read_antenna_response_data


class Telescope:
    """
    Telescope model representation.

    The everybeam telescope is created lazily, once per process, and
    shared by all threads of that process (e.g. dask worker threads).
    Its creation is guarded by a lock. It is excluded from pickling and
    re-created on first use after unpickling.
    """

    def __init__(
        self,
        ms_path: str,
        element_response_model: str,
    ):
        """
        Initialize the Telescope model.

        Parameters
        ----------
        ms_path
            Path to the measurement set.
        element_response_model
            Name of the element response model.
        """
        self._ms_path = ms_path
        self._element_response_model_name = element_response_model
        self._telescope: "eb.Telescope" = None
        self._telescope_lock = threading.Lock()
        self._element_response_model = getattr(
            eb.ElementResponseModel, element_response_model
        )

        self.response_data = read_antenna_response_data(ms_path)
        self._nstations = self.response_data["station_names"].size

    def __dask_tokenize__(self) -> tuple[str, str, str]:
        """
        Deterministic dask token for the telescope.

        The measurement set path and the element response model fully
        identify a telescope, so avoid pickling the response data (and
        the everybeam telescope, if initialised) while tokenizing.

        Returns
        -------
            Tuple identifying this telescope.
        """
        return (
            type(self).__name__,
            str(self._ms_path),
            str(self._element_response_model_name),
        )

    def __getstate__(self) -> dict:
        """
        Pickle state without the everybeam telescope.

        The everybeam telescope is not picklable. It is lazily
        re-created from ``response_data`` after unpickling. This lets an
        initialised telescope be transferred between dask workers.

        Returns
        -------
            Instance state, with the everybeam telescope reset.
        """
        state = self.__dict__.copy()
        state["_telescope"] = None
        del state["_telescope_lock"]
        return state

    def __setstate__(self, state: dict):
        """
        Restore pickled state, with a new lock for lazy initialisation.

        Parameters
        ----------
        state
            Instance state returned by :py:meth:`__getstate__`.
        """
        self.__dict__.update(state)
        self._telescope_lock = threading.Lock()

    def _create_station_node(
        self,
        name: str,
        position: Annotated[npt.NDArray[np.float64], "shape (3,)"],
        coordinate_axes: Annotated[npt.NDArray[np.float64], "shape (3, 3)"],
        element_offsets: Annotated[npt.NDArray[np.float64], "shape (N, 3)"],
        element_flags: Annotated[npt.NDArray[np.bool_], "shape (N, 2)"],
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
        element_flags : array_like, shape (N, 2)
            The flags of the X and Y dipoles of the N elements. A flagged
            dipole does not contribute to the station response.

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
        for offset, (x_flag, y_flag) in zip(
            element_offsets, element_flags, strict=True
        ):
            add_child_element(
                offset,
                is_x_enabled=not x_flag,
                is_y_enabled=not y_flag,
            )

        return station_node

    def _initialize_everybeam_telescope(self):
        """
        Create an OSKAR telescope model containing multiple stations.
        """

        station_names = self.response_data["station_names"]
        positions = self.response_data["positions"]
        coordinate_axes = self.response_data["coordinate_axes"]
        element_offsets = self.response_data["element_offsets"]
        element_flags = self.response_data["element_flags"]
        delay_directions = self.response_data["delay_directions"]

        root_node = eb.StationNode()
        add_child_node = root_node.add_child_node
        for name, pos, axes, offsets, flags in zip(
            station_names,
            positions,
            coordinate_axes,
            element_offsets,
            element_flags,
            strict=True,
        ):
            station_node = self._create_station_node(
                name, pos, axes, offsets, flags
            )
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

        Thread-safe: the telescope is created only once per process,
        even when multiple threads request it concurrently.
        """
        if self._telescope is None:
            with self._telescope_lock:
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
        solution_time: npt.NDArray | np.float64 | float,
        frequencies: npt.NDArray | float | np.float64,
        direction: Annotated[npt.NDArray[np.float64], "shape (3,)"],
        station0: Annotated[npt.NDArray[np.float64], "shape (3,)"],
        scale: Optional[npt.NDArray] = None,
        station_idx: Optional[int] = None,
    ):
        """
        Calculate the station response beams for all stations and frequencies.

        Parameters
        ----------
        solution_time : npt.NDArray or float or np.float64
            Timestamp[s] for the solution.
        frequencies : npt.NDArray or float or np.float64
            Frequency( Or array of frequencies) to compute station response.
        direction : npt.NDArray[np.float64]
            ITRF direction vector in which to evaluate the response, i.e.
            the direction of arrival, shape (3,).
        station0 : npt.NDArray[np.float64]
            ITRF direction vector of the station beam former reference
            (pointing / delay) direction, shape (3,). It is also used as
            the tile beam former reference direction.
        scale : npt.NDArray, optional
            Scaling array per frequency to apply to beams, default is None.
        station_idx: int, optional
            Compute station response for the given station. If station idx
            is not provided, station response is computed for all of the
            stations.

        Returns
        -------
        npt.NDArray[np.complex128]
            Complex beam matrices with shape
            (ntimes, nstations, nfrequencies, 2, 2).
        """

        self._ensure_telescope()
        frequencies = np.atleast_1d(frequencies)
        solution_time = np.atleast_1d(solution_time)

        stations = (
            np.arange(self._nstations)
            if station_idx is None
            else np.array([station_idx])
        )

        # Not locked: the shared everybeam telescope is safe for
        # concurrent use on this path (OSKAR telescope with our own
        # StationNodes). station_response creates a new per-time
        # PhasedArrayPoint for all per-call state, everything below it
        # (station, beamformer, element response) is const, and nothing
        # is lazily filled. The only shared state (NCP cache, ITRF frame,
        # casacore measures tables) is thread-local or mutex-guarded.
        # EveryBeam runs single-threaded here (no OpenMP threads, BLAS or
        # thread pool), so it cannot oversubscribe dask threads.
        # Note: the pybind11 bindings currently hold the GIL, so python
        # threads call this one at a time anyway. Analysed for
        # everybeam v0.8.4 to v0.9.0.
        beams = self._telescope.station_response(
            times=solution_time,
            station_indexes=stations,
            freqs=frequencies,
            direction=direction,
            station0_direction=station0,
            tile0_direction=station0,
        )

        if scale is not None:
            beams *= scale[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]

        return beams
