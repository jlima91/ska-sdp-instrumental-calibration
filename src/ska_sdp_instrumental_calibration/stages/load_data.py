import logging
import math
import os
from typing import Annotated, Literal, Optional, Union

from pydantic import Field
from ska_sdp_piper.piper import CLIArgument, ConfigurableStage
from xradio.measurement_set import convert_msv2_to_processing_set

from ..data_managers.chunking import (
    estimate_channels_per_chunk,
    get_cluster_spec,
)
from ..data_managers.gaintable import create_gaintable_from_msv4
from ..data_managers.processing_set import (
    DataGroup,
    InputType,
    chunk_ms_xdt,
    detect_input_type,
    field_name_without_xradio_suffix,
    get_correlated_data,
    get_field_name,
    input_prefix,
    open_msv4_selection,
    read_msv2_selection_shape,
)
from ..scheduler import UpstreamOutput
from ..tagger import Tags

logger = logging.getLogger(__name__)


@ConfigurableStage(name="load_data")
@Tags.BROADCASTER
def load_data_stage(
    _upstream_output_,
    _output_dir_,
    input_ms: Annotated[list[str], CLIArgument],
    field_id: Annotated[
        int,
        Field(
            description="""MSv2 inputs only: FIELD_ID to calibrate. Ignored
            for Processing Set inputs, which use field_name."""
        ),
    ] = 0,
    ddi_id: Annotated[
        int,
        Field(
            description="""MSv2 inputs only: DATA_DESC_ID to calibrate.
            Ignored for Processing Set inputs, which use spw_name."""
        ),
    ] = 0,
    field_name: Annotated[
        Optional[str],
        Field(
            description="""Processing Set inputs only: name of the field to
            calibrate. If None, each input must hold exactly one field.
            Ignored for MSv2 inputs, which use field_id."""
        ),
    ] = None,
    spw_name: Annotated[
        Optional[str],
        Field(
            description="""Processing Set inputs only: name of the spectral
            window to calibrate. If None, each input must hold exactly one
            spectral window. Ignored for MSv2 inputs, which use ddi_id."""
        ),
    ] = None,
    scan_intents: Annotated[
        Optional[list[str]],
        Field(
            description="""If set, only use data with one of these scan
            intents, e.g. ["CALIBRATE_BANDPASS#ON_SOURCE"]."""
        ),
    ] = None,
    data_group: Annotated[
        DataGroup,
        Field(
            description="""MSv4 data group to calibrate. MSv2 columns DATA,
            CORRECTED_DATA and MODEL_DATA convert to the data groups
            "base", "corrected" and "model"."""
        ),
    ] = "base",
    nchannels_per_chunk: Annotated[
        Union[int, Literal["auto"]],
        Field(
            description="""Number of frequency channels per chunk, used
            across the pipeline and when converting MSv2 inputs. If "auto",
            it is estimated from the visibility size and the threads and
            memory of the dask cluster."""
        ),
    ] = "auto",
) -> list[UpstreamOutput]:
    """
    Load visibilities as MSv4 datasets, one per input.

    The inputs are either MSv2 measurement sets or xradio
    Processing Sets. Each MSv2 is converted to a Processing Set in the
    'processing_sets' sub-directory of the output directory, overwriting
    any previous conversion, keeping only the selected FIELD_ID and
    DATA_DESC_ID, and written with the chunks used across the pipeline.
    Processing Sets are read in place, selecting a field and spectral
    window by name.

    Each input yields one UpstreamOutput, holding a single field, spectral
    window and data group.

    Parameters
    ----------
    _upstream_output_: dict
        Output from the upstream stage
    _output_dir_: str
        Piper builtin. Stores the output directory path.
    input_ms: CLIArgument
        Input MSv2 measurement sets or Processing Sets.
    field_id: int
        MSv2 inputs only: FIELD_ID to calibrate.
    ddi_id: int
        MSv2 inputs only: DATA_DESC_ID to calibrate.
    field_name: str, optional
        Processing Set inputs only: field to calibrate.
    spw_name: str, optional
        Processing Set inputs only: spectral window to calibrate.
    scan_intents: list of str, optional
        Only use data with one of these scan intents.
    data_group: str
        MSv4 data group to calibrate.
    nchannels_per_chunk: int | "auto"
        Number of frequency channels per chunk, used across the pipeline.

    Returns
    -------
    list of UpstreamOutput
        One per input, with the MSv4 node under ``vis``.

    Raises
    ------
    ValueError
        If the inputs mix MSv2 and Processing Sets, or if two inputs share
        an output prefix.
    """
    input_paths = [os.path.realpath(path) for path in input_ms]
    input_type = detect_input_type(input_paths)

    prefixes = [input_prefix(path) for path in input_paths]
    duplicates = sorted({p for p in prefixes if prefixes.count(p) > 1})
    if duplicates:
        raise ValueError(
            f"Inputs must have unique names, found duplicates: {duplicates}"
        )

    return [
        _load_data(
            _output_dir_,
            input_path,
            input_type,
            field_id,
            ddi_id,
            field_name,
            spw_name,
            scan_intents,
            data_group,
            nchannels_per_chunk,
        )
        for input_path in input_paths
    ]


def _load_data(
    _output_dir_: str,
    input_path: str,
    input_type: InputType,
    field_id: int,
    ddi_id: int,
    field_name: Optional[str],
    spw_name: Optional[str],
    scan_intents: Optional[list[str]],
    data_group: DataGroup,
    nchannels_per_chunk: Union[int, Literal["auto"]],
) -> UpstreamOutput:
    """
    Load one input as an MSv4 node, converting it first if it is an MSv2.
    """
    _upstream_output_ = UpstreamOutput()
    prefix = input_prefix(input_path)

    if input_type == "msv2":
        if nchannels_per_chunk == "auto":
            nchannels_per_chunk = _estimate_nchannels_per_chunk(
                read_msv2_selection_shape(input_path, field_id, ddi_id)
            )
        ps_path = os.path.join(
            _output_dir_, "processing_sets", f"{prefix}.ps.zarr"
        )
        logger.info(
            "Converting %s (FIELD_ID %d, DATA_DESC_ID %d) to %s",
            input_path,
            field_id,
            ddi_id,
            ps_path,
        )
        convert_msv2_to_processing_set(
            in_file=input_path,
            out_file=ps_path,
            partition_scheme=["FIELD_ID"],
            partition_filter=lambda partition: (
                field_id in partition["FIELD_ID"]
                and ddi_id in partition["DATA_DESC_ID"]
            ),
            main_chunksize={"frequency": nchannels_per_chunk},
            persistence_mode="w",
        )
        field_name = spw_name = None
    else:
        ps_path = input_path

    ms_xdt = open_msv4_selection(
        ps_path,
        field_name=field_name,
        spw_name=spw_name,
        scan_intents=scan_intents,
        data_group=data_group,
    )

    if nchannels_per_chunk == "auto":
        nchannels_per_chunk = _estimate_nchannels_per_chunk(
            get_correlated_data(ms_xdt).shape
        )

    ms_xdt = chunk_ms_xdt(ms_xdt, nchannels_per_chunk)

    _upstream_output_["ms_prefix"] = prefix
    _upstream_output_["vis"] = ms_xdt
    _upstream_output_["gaintable"] = create_gaintable_from_msv4(
        ms_xdt, "full", "B"
    )
    _upstream_output_["central_beams"] = None
    _upstream_output_["beams_factory"] = None
    # Holds the field name, which names the exported gain tables and their
    # SDM paths, and groups them. The "_<FIELD_ID>" suffix added by xradio
    # (e.g. "3C286_0") is removed, to keep the MSv2 field name ("3C286").
    _upstream_output_["field_id"] = field_name_without_xradio_suffix(
        get_field_name(ms_xdt)
    )
    _upstream_output_["calibration_purpose"] = "bandpass"

    return _upstream_output_


def _estimate_nchannels_per_chunk(vis_shape: tuple[int, ...]) -> int:
    """
    Estimate frequency channels per chunk for visibilities of shape
    (time, baseline, frequency, polarization) on the current dask cluster.
    """
    ntimes, nbaselines, nchannels, npols = vis_shape
    cluster_spec = get_cluster_spec()

    nchannels_per_chunk = estimate_channels_per_chunk(
        nchannels, ntimes * nbaselines * npols, cluster_spec
    )
    logger.info(
        "Auto chunking: %d channels per chunk (%d chunks) for %d channels "
        "on %d threads with %.2f MiB memory per thread",
        nchannels_per_chunk,
        math.ceil(nchannels / nchannels_per_chunk),
        nchannels,
        cluster_spec.total_threads,
        cluster_spec.memory_per_thread_bytes / 2**20,
    )

    return nchannels_per_chunk
