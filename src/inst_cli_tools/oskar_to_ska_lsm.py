#!/usr/bin/env python
"""
Convert an OSKAR sky model file to SKA LSM CSV format.

Two OSKAR formats are read:

1. Named column format, as written by oskar.Sky.save (OSKAR 2.12 and
   later), e.g. the sky_model.csv written by a simulation. The first line
   gives the columns:

   #(RaD,DecD,I,ReferenceFrequency,SpectralIndex,MajorAxis) = format

   Values are separated by commas, and an empty value means "not set". The
   spectral index can be a list of terms, like [-0.7,0.1]. The columns
   which can be written by oskar.Sky.save are supported:

   - RaD, DecD (deg), I (Jy), ReferenceFrequency (Hz)
   - SpectralIndex, and LogarithmicSI (true if not set, as in OSKAR)
   - MajorAxis, MinorAxis (FWHM, arcsec), Orientation (deg)
   - Q, U, V and RotationMeasure, which are dropped

   Other columns raise an error.

2. Old fixed column format, if the first line is not a named column format
   line (a warning is printed):

   RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz),
   Spectral index, Rotation measure (rad/m^2), FWHM major (arcsec),
   FWHM minor (arcsec), Position angle (deg, East of North)

   Fields may be separated by commas and/or whitespace, and anything after
   a '#' is a comment. Only the first 3 columns are required. As in OSKAR,
   rows may have 3-9, 11 (no rotation measure) or 12 columns.

Q, U, V and the rotation measure are not part of the SKA LSM format and are
dropped.

OSKAR stores the full width at half maximum (FWHM) of Gaussian sources,
while the SKA LSM stores semi-axes, so the axes are halved. Both formats
measure the position angle from North through East, so it is unchanged.

The output starts with a CATALOGUE_METADATA_* header, which the pipeline
needs to read the file. The catalogue name defaults to the input file stem.

SKA LSM format columns:
component_id, source_id, ra_deg, dec_deg, i_pol_jy, ref_freq_hz, epoch,
a_arcsec, b_arcsec, pa_deg, spec_idx, log_spec_idx

For more details on the SKA LSM format, please visit:
https://developer.skao.int/projects/ska-sdp-global-sky-model/en/latest/design/lsm-file-structure.html

Usage:
    oskar-to-ska-lsm input.csv
    oskar-to-ska-lsm input.csv output.csv
    oskar-to-ska-lsm input.csv output.csv \
        --comp-prefix GLEAM
    oskar-to-ska-lsm input.csv \
        --default-ref-freq 150e6
    oskar-to-ska-lsm input.csv \
        --catalogue-name MyCat --catalogue-version 2.0.0 --author "A. Person"
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from ska_sdp_datamodels.sky_model import (
    SkyComponent,
    SkyModel,
    SkyModelMetadata,
    write_skymodel_to_lsm_csv,
)
from ska_sdp_datamodels.sky_model.sky_model_io import _tokenize_line

# Columns of the old fixed column format, in order, with their names in the
# named column format
FIXED_COLUMNS = [
    "RaD",  # deg
    "DecD",  # deg
    "I",  # Jy
    "Q",  # Jy
    "U",  # Jy
    "V",  # Jy
    "ReferenceFrequency",  # Hz
    "SpectralIndex",  # one value, or a list like [-0.7,0.1]
    "RotationMeasure",  # rad/m^2
    "MajorAxis",  # FWHM, arcsec
    "MinorAxis",  # FWHM, arcsec
    "Orientation",  # deg, East of North
]

# First line of the named column format, as written by oskar.Sky.save
NAMED_FORMAT_LINE = re.compile(r"^#\((?P<columns>.*)\) = format$")

# Columns accepted in the named column format. LogarithmicSI is true or
# false (true if not set, as in OSKAR). Q, U, V and RotationMeasure are not
# part of the SKA LSM format, and are dropped.
NAMED_COLUMNS = FIXED_COLUMNS + ["LogarithmicSI"]

# Columns of the table returned by the readers
LSM_COLUMNS = [
    "ra_deg",
    "dec_deg",
    "i_pol_jy",
    "ref_freq_hz",
    "spec_idx",
    "log_spec_idx",
    "a_arcsec",
    "b_arcsec",
    "pa_deg",
]


def row_to_lsm(values: Dict[str, str]) -> dict:
    """
    Convert the values of a source, by column name, to the SKA LSM
    convention. Columns which are not given, or empty, are not set: 0, or
    true for LogarithmicSI, as in OSKAR.

    Parameters
    ----------
    values : dict
        Values by name of ``NAMED_COLUMNS``, as in the file.

    Returns
    -------
    dict
        Values of the ``LSM_COLUMNS``.
    """

    def number(column: str) -> float:
        return float(values.get(column) or 0.0)

    spec_idx = values.get("SpectralIndex", "").strip("[]").split(",")
    return {
        "ra_deg": number("RaD"),
        "dec_deg": number("DecD"),
        "i_pol_jy": number("I"),
        "ref_freq_hz": number("ReferenceFrequency"),
        "spec_idx": [float(term) for term in spec_idx if term] or [0.0],
        "log_spec_idx": values.get("LogarithmicSI") != "false",
        # OSKAR stores FWHM, the SKA LSM stores semi-axes
        "a_arcsec": number("MajorAxis") / 2,
        "b_arcsec": number("MinorAxis") / 2,
        "pa_deg": number("Orientation"),
    }


def read_named_columns(
    lines: List[str], columns: List[str], input_file: str
) -> pd.DataFrame:
    """
    Read an OSKAR sky model file in the named column format, as written by
    oskar.Sky.save.

    Parameters
    ----------
    lines : list of str
        Lines of the file.
    columns : list of str
        Column names, from the first line.
    input_file : str
        Path of the file, for error messages.

    Returns
    -------
    pd.DataFrame
        One row per source, with columns ``LSM_COLUMNS``.
    """
    unknown = sorted(set(columns) - set(NAMED_COLUMNS))
    if unknown:
        raise ValueError(
            f"{input_file}: unsupported columns {unknown}. Supported "
            f"columns: {NAMED_COLUMNS}"
        )

    rows = []
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Splits on the commas which are not in a spectral index list
        fields = _tokenize_line(line)
        if len(fields) != len(columns):
            raise ValueError(
                f"{input_file}:{line_num}: {len(fields)} values, but "
                f"{len(columns)} columns"
            )
        rows.append(row_to_lsm(dict(zip(columns, fields))))

    return pd.DataFrame(rows, columns=LSM_COLUMNS)


def read_fixed_columns(lines: List[str], input_file: str) -> pd.DataFrame:
    """
    Read an OSKAR sky model file in the old fixed column format.

    Parameters
    ----------
    lines : list of str
        Lines of the file.
    input_file : str
        Path of the file, for error messages.

    Returns
    -------
    pd.DataFrame
        One row per source, with columns ``LSM_COLUMNS``. Optional
        columns which are not given are set to 0, as in OSKAR.
    """
    rows = []
    for line_num, line in enumerate(lines, start=1):
        fields = re.split(r"[,\s]+", line.split("#", 1)[0].strip())
        fields = [field for field in fields if field]
        if not fields:
            continue

        num_cols = len(fields)
        if num_cols == 11:
            # Legacy format without rotation measure
            fields.insert(8, "0")
        elif num_cols < 3 or num_cols == 10 or num_cols > 12:
            raise ValueError(
                f"{input_file}:{line_num}: expected 3-9, 11 or 12 "
                f"columns, got {num_cols}"
            )
        rows.append(row_to_lsm(dict(zip(FIXED_COLUMNS, fields))))

    return pd.DataFrame(rows, columns=LSM_COLUMNS)


def read_oskar_sky_model(input_file: str) -> pd.DataFrame:
    """
    Read an OSKAR sky model file: in the named column format if the first
    line is a named column format line, else in the fixed column format.

    Parameters
    ----------
    input_file : str
        Path to the input OSKAR sky model file.

    Returns
    -------
    pd.DataFrame
        One row per source, with columns ``LSM_COLUMNS`` (SKA LSM
        convention).
    """
    with open(input_file, encoding="utf-8") as f:
        lines = f.readlines()

    match = NAMED_FORMAT_LINE.match(lines[0].strip()) if lines else None
    if match:
        columns = match["columns"].split(",")
        return read_named_columns(lines, columns, input_file)

    print(
        f"WARNING: the first line of {input_file} is not a named column "
        "format line like '#(RaD,DecD,I) = format'. Reading it as an old "
        "fixed column format file.",
        file=sys.stderr,
    )
    return read_fixed_columns(lines, input_file)


def convert_oskar_to_ska_lsm(
    input_file: str,
    output_file: str,
    component_prefix: str,
    default_ref_freq: Optional[float] = None,
    catalogue_name: Optional[str] = None,
    catalogue_version: str = "1.0.0",
    description: Optional[str] = None,
    author: Optional[str] = None,
):
    """
    Convert an OSKAR sky model file to the SKA LSM CSV format.

    Parameters
    ----------
    input_file : str
        Path to the input OSKAR sky model file (named or fixed column
        format).
    output_file : str
        Path for the output CSV file in SKA LSM format.
    component_prefix : str
        Prefix for generating component IDs.
    default_ref_freq : float, optional
        Reference frequency (Hz) used for components whose reference
        frequency is 0 (the OSKAR default). Their spectral index is set to
        0, as OSKAR applies no spectral scaling to them. If not given, such
        components raise an error.
    catalogue_name : str, optional
        Catalogue name written to the metadata header. Defaults to the
        input file stem.
    catalogue_version : str
        Catalogue version written to the metadata header.
    description : str, optional
        Catalogue description written to the metadata header.
    author : str, optional
        Catalogue author written to the metadata header.
    """
    df = read_oskar_sky_model(input_file)

    zero_ref_freq = df["ref_freq_hz"] == 0
    if zero_ref_freq.any():
        if default_ref_freq is None:
            raise ValueError(
                f"{zero_ref_freq.sum()} components in {input_file} have a "
                "reference frequency of 0 Hz. Pass --default-ref-freq to "
                "use a fixed reference frequency for them instead."
            )
        print(
            f"Using a reference frequency of {default_ref_freq} Hz for "
            f"{zero_ref_freq.sum()} components with 0 Hz, and setting "
            "their spectral index to 0."
        )
        df.loc[zero_ref_freq, "ref_freq_hz"] = default_ref_freq
        # OSKAR applies no spectral scaling when the reference frequency is
        # 0, so these components must stay flat-spectrum.
        df["spec_idx"] = [
            [0.0] if is_zero else spec_idx
            for is_zero, spec_idx in zip(zero_ref_freq, df["spec_idx"])
        ]

    df["component_id"] = f"{component_prefix} " + df.index.astype(
        str
    ).str.zfill(6)
    df["source_id"] = ""
    df["epoch"] = 0.0

    components = {
        row["component_id"]: SkyComponent(**row)
        for row in df.to_dict("records")
    }
    sky_model = SkyModel(
        metadata=SkyModelMetadata(
            version=catalogue_version,
            catalogue_name=catalogue_name or Path(input_file).stem,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            author=author,
        ),
        components=components,
    )
    write_skymodel_to_lsm_csv(sky_model, output_file)

    print(f"Converted {len(df)} components from {input_file} to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an OSKAR sky model file (named or fixed column "
        "format) to SKA LSM CSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input OSKAR sky model file.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=str,
        default="./converted_sky_model.csv",
        help="Path for the output CSV file in SKA LSM format.",
    )
    parser.add_argument(
        "--comp-prefix",
        type=str,
        default="Component",
        help="Prefix for generating component IDs. Default: Component",
    )

    parser.add_argument(
        "--default-ref-freq",
        type=float,
        default=None,
        help="Reference frequency (Hz) to use for components whose "
        "reference frequency is 0. Without it, such components cause an "
        "error.",
    )
    parser.add_argument(
        "--catalogue-name",
        type=str,
        default=None,
        help="Catalogue name for the metadata header. "
        "Default: the input file stem",
    )
    parser.add_argument(
        "--catalogue-version",
        type=str,
        default="1.0.0",
        help="Catalogue version for the metadata header. Default: 1.0.0",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Catalogue description for the metadata header.",
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Catalogue author for the metadata header.",
    )

    args = parser.parse_args()

    convert_oskar_to_ska_lsm(
        input_file=args.input_csv,
        output_file=args.output_csv,
        component_prefix=args.comp_prefix,
        default_ref_freq=args.default_ref_freq,
        catalogue_name=args.catalogue_name,
        catalogue_version=args.catalogue_version,
        description=args.description,
        author=args.author,
    )


if __name__ == "__main__":
    main()
