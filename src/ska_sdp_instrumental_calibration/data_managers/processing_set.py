"""
Reading visibilities as MSv4 (xradio) datasets.

INST accepts either MSv2 measurement sets or xradio Processing Sets (PS) as
input. MSv2 inputs are converted to a PS first, so the rest of the pipeline
only deals with the MSv4 schema. Each input is reduced to a single MSv4 node
(an ``xr.DataTree``) holding one field, one spectral window and one data
group.
"""

import logging
import os
import re
from typing import Literal, Optional, Sequence

import astropy.units as u
import numpy as np
import xarray as xr
import zarr
from astropy.coordinates import EarthLocation, SkyCoord
from casacore.tables import table
from xradio.measurement_set import open_processing_set

logger = logging.getLogger(__name__)

# Seconds between the MJD epoch (1858-11-17) and the Unix epoch (1970-01-01),
# i.e. 40587 days. xradio subtracts exactly this from the MSv2 TIME column
# (casacore MJD seconds, UTC) to get the MSv4 time (Unix seconds, UTC).
MJD_UNIX_OFFSET_SECONDS = 3506716800.0

InputType = Literal["msv2", "ps"]

DataGroup = Literal["base", "corrected", "model"]

_INPUT_SUFFIXES = (".ps.zarr", ".zarr", ".ps", ".ms")

# xradio suffixes MSv2 field names with "_<FIELD_ID>" to make them unique
_XRADIO_FIELD_ID_SUFFIX = re.compile(r"_\d+$")

# Name used for a field without name, as by the MSv2 Visibility reader
UNKNOWN_FIELD_NAME = "UNKNOWN_FIELD"


def input_prefix(path: str) -> str:
    """
    Output prefix of an input, i.e. its basename without a known suffix.

    Parameters
    ----------
    path
        Path to an MSv2 or Processing Set.

    Returns
    -------
        For example ``obs1`` for ``/data/obs1.ms`` or ``/data/obs1.ps.zarr``.
    """
    name = os.path.basename(os.path.normpath(path))
    for suffix in _INPUT_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def is_processing_set(path: str) -> bool:
    """
    Whether ``path`` is a zarr store holding an xradio Processing Set.

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
        True if the root group of the store has ``type: processing_set``.
    """
    if not os.path.isdir(path):
        return False
    try:
        attrs = zarr.open_group(path, mode="r").attrs
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return attrs.get("type") == "processing_set"


def is_msv2(path: str) -> bool:
    """
    Whether ``path`` is a casacore table, i.e. an MSv2 measurement set.

    Parameters
    ----------
    path
        Path to check.

    Returns
    -------
        True if ``path`` is a directory containing ``table.dat``.
    """
    return os.path.isfile(os.path.join(path, "table.dat"))


def detect_input_type(paths: Sequence[str]) -> InputType:
    """
    Detect whether the inputs are all MSv2 or all Processing Sets.

    Parameters
    ----------
    paths
        Input paths.

    Returns
    -------
        ``"msv2"`` or ``"ps"``.

    Raises
    ------
    ValueError
        If there are no inputs, if an input is neither an MSv2 nor a
        Processing Set, or if the inputs mix both types.
    """
    if not paths:
        raise ValueError("No input visibilities given")

    types = set()
    for path in paths:
        if is_processing_set(path):
            types.add("ps")
        elif is_msv2(path):
            types.add("msv2")
        else:
            raise ValueError(
                f"Input {path} is neither an MSv2 measurement set nor an "
                "xradio Processing Set"
            )

    if len(types) > 1:
        raise ValueError(
            "Inputs mix MSv2 measurement sets and Processing Sets. "
            "Give either only MSv2 inputs or only Processing Set inputs."
        )
    return types.pop()


def read_msv2_selection_shape(
    ms_path: str, field_id: int, ddi_id: int
) -> tuple[int, int, int, int]:
    """
    Approximate visibility shape of one field and DDI of an MSv2.

    Used to choose the frequency chunk size before converting to a
    Processing Set, so that the converted store is written with the
    chunking the pipeline will use.

    Parameters
    ----------
    ms_path
        Path to the MSv2.
    field_id
        MSv2 FIELD_ID to select.
    ddi_id
        MSv2 DATA_DESC_ID to select.

    Returns
    -------
        ``(ntimes, nbaselines, nchannels, npols)``. The number of baselines
        is the average number of rows per time step.

    Raises
    ------
    ValueError
        If ``ddi_id`` does not exist, or no rows match the selection.
    """
    with table(f"{ms_path}/DATA_DESCRIPTION", ack=False) as tb:
        if not 0 <= ddi_id < tb.nrows():
            raise ValueError(
                f"DATA_DESC_ID {ddi_id} not found in {ms_path}, which has "
                f"{tb.nrows()} data descriptions"
            )
        spw_id = tb.getcell("SPECTRAL_WINDOW_ID", ddi_id)
        pol_id = tb.getcell("POLARIZATION_ID", ddi_id)

    with table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as tb:
        nchannels = int(tb.getcell("NUM_CHAN", spw_id))

    with table(f"{ms_path}/POLARIZATION", ack=False) as tb:
        npols = int(tb.getcell("NUM_CORR", pol_id))

    with table(ms_path, ack=False) as tb:
        selected = tb.query(
            f"FIELD_ID=={field_id} AND DATA_DESC_ID=={ddi_id}",
            columns="TIME",
        )
        try:
            nrows = selected.nrows()
            ntimes = len(np.unique(selected.getcol("TIME"))) if nrows else 0
        finally:
            selected.close()

    if nrows == 0:
        raise ValueError(
            f"No rows with FIELD_ID {field_id} and DATA_DESC_ID {ddi_id} "
            f"in {ms_path}"
        )

    return ntimes, nrows // ntimes, nchannels, npols


def _resolve_single(
    kind: str, requested: Optional[str], available: set[str], path: str
) -> str:
    """
    Pick exactly one name from ``available``.

    If ``requested`` is None, there must be exactly one available name.
    Otherwise ``requested`` must be one of them.
    """
    choices = sorted(available)
    if requested is None:
        if len(choices) != 1:
            raise ValueError(
                f"{path} contains {len(choices)} {kind}s {choices}. "
                f"Set {kind}_name to select one."
            )
        return choices[0]

    if requested not in available:
        raise ValueError(
            f"{kind}_name {requested!r} not found in {path}. "
            f"Available {kind}s: {choices}"
        )
    return requested


def _select_field_times(ms_xdt: xr.DataTree, field_name: str) -> xr.DataTree:
    """
    Keep only the time steps of ``field_name`` in an MSv4 node.

    A node may hold several fields along time when the Processing Set was
    not partitioned by field.
    """
    in_field = np.asarray(ms_xdt.ds.field_name.values) == field_name
    if in_field.all():
        return ms_xdt

    selected = ms_xdt.copy(deep=False)
    selected.ds = ms_xdt.ds.isel(time=np.flatnonzero(in_field))
    return selected


def _concat_along_time(nodes: list[xr.DataTree]) -> xr.DataTree:
    """
    Concatenate MSv4 nodes of the same field and spectral window along time.

    The sub-datasets of the first node are kept. All nodes must share the
    same baselines and frequencies.
    """
    if len(nodes) == 1:
        return nodes[0]

    first = nodes[0].ds
    for node in nodes[1:]:
        for coord in (
            "baseline_antenna1_name",
            "baseline_antenna2_name",
            "frequency",
        ):
            if not np.array_equal(first[coord].values, node.ds[coord].values):
                raise ValueError(
                    f"Cannot combine MSv4 nodes {nodes[0].name} and "
                    f"{node.name}: their {coord} differ"
                )

    combined_ds = xr.concat(
        [node.to_dataset() for node in nodes],
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    ).sortby("time")

    combined = nodes[0].copy(deep=False)
    combined.ds = combined_ds
    return combined


def open_msv4_selection(
    ps_path: str,
    field_name: Optional[str] = None,
    spw_name: Optional[str] = None,
    scan_intents: Optional[list[str]] = None,
    data_group: DataGroup = "base",
) -> xr.DataTree:
    """
    Open a Processing Set and select a single field and spectral window.

    All MSv4 nodes holding the selected field and spectral window are
    combined along time into one node, keeping only the time steps of the
    selected field and only the variables of the selected data group.

    Parameters
    ----------
    ps_path
        Path to the Processing Set.
    field_name
        Field to select. If None, the Processing Set must hold exactly one
        field in the selected spectral window.
    spw_name
        Spectral window to select. If None, the Processing Set must hold
        exactly one spectral window.
    scan_intents
        If given, only MSv4 nodes with one of these scan intents are used.
    data_group
        Data group to use, e.g. ``base``, ``corrected`` or ``model``.

    Returns
    -------
        The selected MSv4 node, opened lazily and not rechunked. Its field
        is given by :func:`get_field_name`.

    Raises
    ------
    ValueError
        If nothing matches the selection, if the selection is ambiguous,
        or if the data group is missing.
    """
    ps_xdt = open_processing_set(ps_path, scan_intents=scan_intents)
    if not ps_xdt.children:
        raise ValueError(
            f"No MSv4 data in {ps_path} matches scan_intents {scan_intents}"
        )

    summary = ps_xdt.xr_ps.summary()
    spw_name = _resolve_single(
        "spw", spw_name, set(summary["spw_name"]), ps_path
    )
    summary = summary[summary["spw_name"] == spw_name]

    fields = {name for names in summary["field_name"] for name in names}
    field_name = _resolve_single("field", field_name, fields, ps_path)

    node_names = [
        row["name"]
        for _, row in summary.iterrows()
        if field_name in row["field_name"]
    ]

    nodes = []
    for node_name in node_names:
        node = ps_xdt[node_name]
        available_groups = list(node.attrs["data_groups"])
        if data_group not in available_groups:
            raise ValueError(
                f"Data group {data_group!r} not found in {ps_path}/"
                f"{node_name}. Available data groups: {available_groups}"
            )
        node = node.xr_ms.sel(data_group_name=data_group)
        nodes.append(_select_field_times(node, field_name))

    logger.info(
        "Selected field %s, spw %s, data group %s from %s (MSv4 nodes: %s)",
        field_name,
        spw_name,
        data_group,
        ps_path,
        node_names,
    )

    return _concat_along_time(nodes)


def chunk_ms_xdt(ms_xdt: xr.DataTree, nchannels_per_chunk: int) -> xr.DataTree:
    """
    Rechunk an MSv4 node with the chunks used across the pipeline: every
    dimension whole, except frequency.

    xradio opens a Processing Set with the chunks it is stored with, which
    for Processing Sets not converted by INST can be any. Coordinates along
    time (e.g. ``field_name``) are rechunked too, which also unifies the
    chunks xradio leaves inconsistent between coordinates and data
    variables on opening. Data already stored with these chunks (MSv2
    inputs, converted by load_data) is not rechunked.

    Parameters
    ----------
    ms_xdt
        MSv4 node.
    nchannels_per_chunk
        Number of channels per chunk.

    Returns
    -------
        Shallow copy of the node with rechunked visibilities.
    """
    ds = ms_xdt.to_dataset()
    chunks = {dim: -1 for dim in ds.dims}
    chunks["frequency"] = nchannels_per_chunk
    chunked = ms_xdt.copy(deep=False)
    chunked.ds = ds.chunk(chunks)
    return chunked


def get_field_name(ms_xdt: xr.DataTree) -> str:
    """
    Name of the single field of an MSv4 node, as written by xradio.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single field, e.g. from
        :func:`open_msv4_selection`.

    Returns
    -------
        Field name, e.g. ``3C286_0``.
    """
    return str(np.asarray(ms_xdt.ds.field_name.values)[0])


def field_name_without_xradio_suffix(field_name: str) -> str:
    """
    Field name of an MSv4 node, without the ``_<FIELD_ID>`` suffix that
    xradio appends to MSv2 field names on conversion.

    For data converted from MSv2, this is the MSv2 ``FIELD.NAME``, which
    names the exported gaintables and their SDM directories, as before the
    MSv4 migration. Any trailing ``_<digits>`` is removed, so a field name
    written by another tool that ends so is truncated as well.

    Parameters
    ----------
    field_name
        MSv4 field name, e.g. ``3C286_0``.

    Returns
    -------
        Field name without the suffix, e.g. ``3C286``, or
        ``UNKNOWN_FIELD`` if that is empty (unnamed MSv2 field).
    """
    return _XRADIO_FIELD_ID_SUFFIX.sub("", field_name) or UNKNOWN_FIELD_NAME


def msv4_time_to_mjd_seconds(time: float | np.ndarray) -> float | np.ndarray:
    """
    Convert MSv4 times to MSv2 times.

    MSv4 times are UTC seconds since the Unix epoch. The beam and sky model
    code (and EveryBeam) works in UTC seconds since the MJD epoch, as in the
    MSv2 TIME column and the ska-sdp-datamodels Visibility.

    Parameters
    ----------
    time
        MSv4 time(s), e.g. the ``time`` coordinate of an MSv4 node or of a
        gaintable created from it.

    Returns
    -------
        The same time(s) in seconds since the MJD epoch.
    """
    return np.asarray(time) + MJD_UNIX_OFFSET_SECONDS


def get_data_group(ms_xdt: xr.DataTree) -> dict[str, str]:
    """
    The single data group of an MSv4 node.

    Parameters
    ----------
    ms_xdt
        MSv4 node selected with a single data group, e.g. from
        :func:`open_msv4_selection`.

    Returns
    -------
        Data group, mapping roles (``correlated_data``, ``flag``,
        ``weight``, ``uvw``, ...) to variable names.

    Raises
    ------
    ValueError
        If the node does not hold exactly one data group.
    """
    data_groups = ms_xdt.attrs["data_groups"]
    if len(data_groups) != 1:
        raise ValueError(
            f"Expected a single data group in MSv4 node {ms_xdt.name}, "
            f"found {list(data_groups)}"
        )
    (group,) = data_groups.values()
    return group


def get_correlated_data(ms_xdt: xr.DataTree) -> xr.DataArray:
    """
    Correlated data of the single data group of an MSv4 node.

    Replaces ``Visibility.vis``.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single data group.

    Returns
    -------
        Correlated data, e.g. ``VISIBILITY``, dims
        ``(time, baseline_id, frequency, polarization)``.
    """
    return ms_xdt.ds[get_data_group(ms_xdt)["correlated_data"]]


def with_correlated_data(
    ms_xdt: xr.DataTree, correlated_data: xr.DataArray
) -> xr.DataTree:
    """
    Copy of an MSv4 node with its correlated data replaced.

    Replaces ``vis.assign({"vis": ...})``. Other variables (flags, weights,
    UVW, ...) and child datasets are shared with ``ms_xdt``.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single data group.
    correlated_data
        New correlated data, with the same dims and dimension coordinates
        as the current one, in any dimension order.

    Returns
    -------
        Shallow copy of the node, with the attributes and non-dimension
        coordinates of the current correlated data.

    Raises
    ------
    ValueError
        If the dims or dimension coordinate values differ.
    """
    current = get_correlated_data(ms_xdt)
    if set(correlated_data.dims) != set(current.dims):
        raise ValueError(
            f"Correlated data dims {correlated_data.dims} do not match "
            f"{current.dims}"
        )
    correlated_data = correlated_data.transpose(*current.dims)
    for dim in current.dims:
        if dim in correlated_data.coords and not np.array_equal(
            correlated_data[dim].values, current[dim].values
        ):
            raise ValueError(f"Correlated data {dim} coordinate differs")

    replaced = ms_xdt.copy(deep=False)
    replaced.ds = ms_xdt.to_dataset().assign(
        {current.name: current.copy(data=correlated_data.data)}
    )
    return replaced


def _phase_centre_direction(ms_xdt: xr.DataTree) -> xr.DataArray:
    """
    Phase centre direction of the single field of an MSv4 node, as
    ``(ra, dec)`` in radians.

    Raises
    ------
    NotImplementedError
        If the phase centre varies with time (e.g. ephemeris fields), or is
        not given in radians in an FK5 or ICRS frame.
    """
    field_and_source = ms_xdt.xr_ms.get_field_and_source_xds()
    direction = field_and_source.FIELD_PHASE_CENTER_DIRECTION

    if "field_name" not in direction.dims:
        raise NotImplementedError(
            "Time dependent phase centres (e.g. ephemeris fields) are not "
            "supported"
        )
    direction = direction.sel(field_name=get_field_name(ms_xdt))

    units, frame = direction.attrs.get("units"), direction.attrs.get("frame")
    if units != "rad" or str(frame).lower() not in ("fk5", "icrs"):
        raise NotImplementedError(
            f"Unsupported phase centre with units {units!r} and frame "
            f"{frame!r}. Expected radians in an FK5 or ICRS frame."
        )
    return direction.transpose("sky_dir_label")


def get_phase_centre(ms_xdt: xr.DataTree) -> SkyCoord:
    """
    Phase centre of the single field of an MSv4 node.

    Replaces ``Visibility.phasecentre``.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single field, e.g. from
        :func:`open_msv4_selection`.

    Returns
    -------
        Phase centre direction.
    """
    ra, dec = _phase_centre_direction(ms_xdt).values
    # xradio labels MSv2 PHASE_DIR (J2000) as FK5. It is read as ICRS, like
    # the MSv2 Visibility reader does, so that results match the MSv2 path.
    return SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")


def get_array_location(ms_xdt: xr.DataTree) -> EarthLocation:
    """
    Array location of an MSv4 node, i.e. the position of its first antenna.

    Replaces ``Visibility.configuration.location``. The first antenna is
    used, as in the MSv2 Visibility reader. The OSKAR beam normalisation in
    :class:`~.beams.BeamsLow` relies on this.

    Parameters
    ----------
    ms_xdt
        MSv4 node with an ``antenna_xds`` child.

    Returns
    -------
        Geocentric location of the first antenna.

    Raises
    ------
    NotImplementedError
        If antenna positions are not geocentric ITRS positions in metres.
    """
    positions = ms_xdt["antenna_xds"].ds.ANTENNA_POSITION
    units, frame = positions.attrs.get("units"), positions.attrs.get("frame")
    if units != "m" or frame != "ITRS":
        raise NotImplementedError(
            f"Unsupported antenna positions with units {units!r} and frame "
            f"{frame!r}. Expected ITRS positions in metres."
        )
    x, y, z = positions.transpose("antenna_name", "cartesian_pos_label")[
        0
    ].values
    return EarthLocation.from_geocentric(x, y, z, unit="m")


def get_baseline_antenna_indices(
    ms_xdt: xr.DataTree,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Antenna indices of each baseline of an MSv4 node.

    Replaces ``Visibility.antenna1`` and ``Visibility.antenna2``. Indices
    are positions in ``antenna_xds.antenna_name``, which is also the antenna
    order of gaintables and of the EveryBeam telescope. Baselines keep the
    antenna order of the input data, which may have antenna1 > antenna2.

    Parameters
    ----------
    ms_xdt
        MSv4 node with an ``antenna_xds`` child.

    Returns
    -------
        ``(antenna1, antenna2)`` index arrays of shape ``(baseline_id,)``.
    """
    antenna_names = ms_xdt["antenna_xds"].ds.antenna_name.values
    index = {name: i for i, name in enumerate(antenna_names)}
    ds = ms_xdt.ds
    return tuple(
        np.array([index[name] for name in ds[coord].values], dtype=np.int64)
        for coord in ("baseline_antenna1_name", "baseline_antenna2_name")
    )


def get_uvw_for_prediction(ms_xdt: xr.DataTree) -> xr.DataArray:
    """
    UVW of an MSv4 node, in the sign convention of the prediction code.

    MSv4 keeps the UVW of the MSv2 as is. The ska-sdp-datamodels Visibility
    negates it (see ``_load_uvw_xdr`` in visibility.py), and
    ``dft_skycomponent`` expects that convention. So the UVW is negated here.

    The MSv2 Visibility reader also swaps baselines with antenna1 > antenna2,
    flipping their UVW sign back and conjugating their visibilities. MSv4
    does not reorder baselines, so negating every baseline is consistent
    with the unmodified visibilities and the antenna indices from
    :func:`get_baseline_antenna_indices`.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single data group.

    Returns
    -------
        Negated UVW, dims ``(time, baseline_id, uvw_label)``.
    """
    uvw = ms_xdt.ds[get_data_group(ms_xdt)["uvw"]]
    return (-uvw).assign_attrs(uvw.attrs)


def get_delay_directions(ms_xdt: xr.DataTree) -> np.ndarray:
    """
    EveryBeam delay direction of the single field of an MSv4 node.

    The MSv2 FIELD DELAY_DIR is not converted by xradio, so the phase
    centre is used. They are equal unless the visibilities were phase
    rotated after correlation.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single field.

    Returns
    -------
        ``(ra, dec)`` in radians.
    """
    return _phase_centre_direction(ms_xdt).values


def read_station_layout_from_msv4(
    ms_xdt: xr.DataTree,
) -> dict[str, np.ndarray]:
    """
    Read the station layout needed by EveryBeam from an MSv4 node.

    Unlike :func:`read_antenna_response_data_from_msv4`, this does not
    depend on the field, so any MSv4 node of a Processing Set may be used.

    Parameters
    ----------
    ms_xdt
        MSv4 node with ``antenna_xds`` and ``phased_array_xds`` children.

    Returns
    -------
        Dictionary of numpy arrays with station names, positions,
        coordinate axes, element offsets of shape (nstations, nelements, 3)
        and element flags of shape (nstations, nelements, 2).

    Raises
    ------
    ValueError
        If the node has no ``phased_array_xds``.
    """
    if "phased_array_xds" not in ms_xdt.children:
        raise ValueError(
            f"MSv4 node {ms_xdt.name} has no phased_array_xds, which is "
            "required to create the EveryBeam telescope"
        )

    antenna = ms_xdt["antenna_xds"].ds
    phased_array = (
        ms_xdt["phased_array_xds"]
        .ds.sel(antenna_name=antenna.antenna_name)
        .transpose("antenna_name", "element_id", ...)
    )

    return dict(
        station_names=antenna.antenna_name.values,
        positions=antenna.ANTENNA_POSITION.values,
        coordinate_axes=phased_array.PHASED_ARRAY_COORDINATE_AXES.values,
        element_offsets=phased_array.PHASED_ARRAY_ELEMENT_OFFSET.values,
        element_flags=phased_array.PHASED_ARRAY_ELEMENT_FLAG.values,
    )


def read_antenna_response_data_from_msv4(
    ms_xdt: xr.DataTree,
) -> dict[str, np.ndarray]:
    """
    Read the antenna response data needed by EveryBeam from an MSv4 node.

    MSv4 equivalent of
    :func:`~.visibility.read_antenna_response_data`, returning the same
    dictionary: the :func:`read_station_layout_from_msv4` of the node, with
    the :func:`get_delay_directions` of its field.

    Parameters
    ----------
    ms_xdt
        MSv4 node holding a single field, with ``antenna_xds`` and
        ``phased_array_xds`` children.

    Returns
    -------
        Dictionary of numpy arrays with station names, positions, delay
        directions, coordinate axes, element offsets and element flags.

    Raises
    ------
    ValueError
        If the node has no ``phased_array_xds``.
    """
    return dict(
        read_station_layout_from_msv4(ms_xdt),
        delay_directions=get_delay_directions(ms_xdt),
    )
