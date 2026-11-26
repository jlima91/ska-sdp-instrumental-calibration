# pylint:disable=c-extension-no-member
import threading
from typing import Annotated, Optional

import everybeam as eb
import numpy as np
import numpy.typing as npt
import xarray as xr
from dask.base import tokenize

from xradio.measurement_set import open_processing_set

from .processing_set import (
    detect_input_type,
    get_delay_directions,
    read_antenna_response_data_from_msv4,
    read_station_layout_from_msv4,
)
from .visibility import read_antenna_response_data


class Telescope:
    """
    Telescope model representation.

    The everybeam telescope
    is created lazily, once per process, and
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
        Initialize the Telescope model from an MSv2 measurement set.

        Parameters
        ----------
        ms_path
            Path to the measurement set.
        element_response_model
            Name of the element response model.
        """
        self._setup(
            identity=str(ms_path),
            element_response_model=element_response_model,
            response_data=read_antenna_response_data(ms_path),
        )

    @classmethod
    def from_msv4(
        cls,
        ms_xdt: xr.DataTree,
        element_response_model: str,
    ) -> "Telescope":
        """
        Create the Telescope model from an MSv4 node.

        Parameters
        ----------
        ms_xdt
            MSv4 node holding a single field, with ``antenna_xds`` and
            ``phased_array_xds`` children.
        element_response_model
            Name of the element response model.

        Returns
        -------
            Telescope with the stations of ``antenna_xds``, in that order.
        """
        return cls._from_response_data(
            read_antenna_response_data_from_msv4(ms_xdt),
            element_response_model,
        )

    @classmethod
    def for_visibilities(
        cls,
        ms_xdt: xr.DataTree,
        element_response_model: str,
        eb_ms: Optional[str] = None,
    ) -> "Telescope":
        """
        Create the Telescope model for the visibilities of an MSv4 node.

        Only the beam metadata (antennas, phased arrays and field direction)
        is read, eagerly and once, on the client. It is small and does not
        depend on the visibilities.

        The beam always points at the field of ``ms_xdt``: only the station
        layout is read from ``eb_ms``, so its fields and spectral windows do
        not matter.

        Parameters
        ----------
        ms_xdt
            MSv4 node being calibrated.
        element_response_model
            Name of the element response model.
        eb_ms
            Optional MSv2 or Processing Set to read the station layout
            from, instead of ``ms_xdt``. An MSv2 is read directly from its
            subtables, without converting it to a Processing Set.

        Returns
        -------
            Telescope with stations in the ``antenna_xds`` order of
            ``ms_xdt``.

        Raises
        ------
        ValueError
            If the stations of ``eb_ms`` differ from the antennas of
            ``ms_xdt``, as beams are matched to antennas by index.
        """
        if eb_ms is None:
            return cls.from_msv4(ms_xdt, element_response_model)

        response_data = _read_station_layout(eb_ms)
        antenna_names = ms_xdt["antenna_xds"].ds.antenna_name.values
        if not np.array_equal(response_data["station_names"], antenna_names):
            raise ValueError(
                f"Stations of eb_ms {eb_ms} differ from the antennas of the "
                "visibilities. They must have the same names in the same "
                "order."
            )
        response_data["delay_directions"] = get_delay_directions(ms_xdt)
        return cls._from_response_data(response_data, element_response_model)

    @classmethod
    def _from_response_data(
        cls,
        response_data: dict[str, np.ndarray],
        element_response_model: str,
    ) -> "Telescope":
        """
        Create the Telescope model from antenna response data already read,
        identified by its hash for dask tokenization.
        """
        telescope = cls.__new__(cls)
        telescope._setup(
            identity=tokenize(response_data),
            element_response_model=element_response_model,
            response_data=response_data,
        )
        return telescope

    def _setup(
        self,
        identity: str,
        element_response_model: str,
        response_data: dict[str, np.ndarray],
    ):
        """
        Set the instance state shared by all constructors.

        Parameters
        ----------
        identity
            String which, with the element response model, uniquely
            identifies the telescope. Used for dask tokenization.
        element_response_model
            Name of the element response model.
        response_data
            Antenna response data, see
            :func:`~.visibility.read_antenna_response_data`.
        """
        self._identity = identity
        self._element_response_model_name = element_response_model
        self._telescope: "eb.Telescope" = None
        self._telescope_lock = threading.Lock()
        self._element_response_model = getattr(
            eb.ElementResponseModel, element_response_model
        )

        self.response_data = response_data
        self._nstations = self.response_data["station_names"].size

    def __dask_tokenize__(self) -> tuple[str, str, str]:
        """
        Deterministic dask token for the telescope.

        The identity (measurement set path, or hash of the MSv4 response
        data) and the element response model fully identify a telescope,
        so avoid pickling the response data (and the everybeam telescope,
        if initialised) while tokenizing.

        Returns
        -------
            Tuple identifying this telescope.
        """
        return (
            type(self).__name__,
            self._identity,
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


def _read_station_layout(eb_ms: str) -> dict[str, np.ndarray]:
    """
    Station layout of an MSv2 or Processing Set, i.e. its antenna response
    data without delay directions.

    Parameters
    ----------
    eb_ms
        Path to an MSv2 or a Processing Set.

    Returns
    -------
        See :func:`~.processing_set.read_station_layout_from_msv4`.

    Raises
    ------
    ValueError
        If the Processing Set holds no MSv4 node.
    """
    if detect_input_type([eb_ms]) == "msv2":
        response_data = read_antenna_response_data(eb_ms)
        # DELAY_DIR holds one row per field, of any field of the MS
        del response_data["delay_directions"]
        return response_data

    # The layout does not depend on the field or spectral window, so the
    # first MSv4 node is used
    nodes = iter(open_processing_set(eb_ms).children.values())
    first_node = next(nodes, None)
    if first_node is None:
        raise ValueError(f"No MSv4 data in eb_ms {eb_ms}")
    return read_station_layout_from_msv4(first_node)
