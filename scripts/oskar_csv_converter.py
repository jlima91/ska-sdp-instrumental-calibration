#!/usr/bin/env python
"""
Convert OSKAR CSV format to SKA LSM CSV format.

Old OSKAR format columns:
RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz),
Spectral index, Rotation measure (rad/m^2), FWHM major (arcsec),
FWHM minor (arcsec), Position angle (deg, East of North)

Fields may be separated by commas and/or whitespace, and anything after
a '#' is a comment. Only the first 3 columns are required. As in OSKAR,
rows may have 3-9, 11 (no rotation measure) or 12 columns. Q, U, V and
the rotation measure are not part of the SKA LSM format and are dropped.

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
    python scripts/oskar_csv_converter.py input.csv
    python scripts/oskar_csv_converter.py input.csv output.csv
    python scripts/oskar_csv_converter.py input.csv output.csv --comp-prefix GLEAM
    python scripts/oskar_csv_converter.py input.csv --default-ref-freq 150e6
    python scripts/oskar_csv_converter.py input.csv --catalogue-name MyCat \
        --catalogue-version 2.0.0 --author "A. Person"
"""

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from ska_sdp_datamodels.sky_model import (
    SkyComponent,
    SkyModel,
    SkyModelMetadata,
    write_skymodel_to_lsm_csv,
)

# Old OSKAR CSV headers
OSKAR_HEADERS = [
    "RA (deg)",
    "Dec (deg)",
    "I (Jy)",
    "Q (Jy)",
    "U (Jy)",
    "V (Jy)",
    "Ref. freq. (Hz)",
    "Spectral index",
    "Rotation measure (rad/m^2)",
    "FWHM major (arcsec)",
    "FWHM minor (arcsec)",
    "Position angle (deg)",
]


def read_oskar_csv(input_file: str) -> pd.DataFrame:
    """
    Read an OSKAR fixed format sky model file.

    Parameters
    ----------
    input_file : str
        Path to the input OSKAR sky model file.

    Returns
    -------
    pd.DataFrame
        One row per source, with columns ``OSKAR_HEADERS``. Optional
        columns which are not given are set to 0, as in OSKAR.
    """
    rows = []
    with open(input_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            fields = re.split(r"[,\s]+", line.split("#", 1)[0].strip())
            fields = [field for field in fields if field]
            if not fields:
                continue

            num_cols = len(fields)
            if num_cols == 11:
                # Legacy format without rotation measure
                fields.insert(8, 0.0)
            elif num_cols < 3 or num_cols == 10 or num_cols > 12:
                raise ValueError(
                    f"{input_file}:{line_num}: expected 3-9, 11 or 12 "
                    f"columns, got {num_cols}"
                )

            values = [float(field) for field in fields]
            rows.append(values + [0.0] * (len(OSKAR_HEADERS) - len(values)))

    return pd.DataFrame(rows, columns=OSKAR_HEADERS, dtype=float)


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
    Convert an OSKAR format CSV file to the SKA LSM CSV format.

    Parameters
    ----------
    input_file : str
        Path to the input OSKAR CSV file.
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
    df = read_oskar_csv(input_file)

    zero_ref_freq = df["Ref. freq. (Hz)"] == 0
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
        df.loc[zero_ref_freq, "Ref. freq. (Hz)"] = default_ref_freq
        # OSKAR applies no spectral scaling when the reference frequency is
        # 0, so these components must stay flat-spectrum.
        df.loc[zero_ref_freq, "Spectral index"] = 0.0

    header_mapping = {
        "RA (deg)": "ra_deg",
        "Dec (deg)": "dec_deg",
        "I (Jy)": "i_pol_jy",
        "FWHM major (arcsec)": "a_arcsec",
        "FWHM minor (arcsec)": "b_arcsec",
        "Position angle (deg)": "pa_deg",
        "Ref. freq. (Hz)": "ref_freq_hz",
        "Spectral index": "spec_idx",
    }
    df = df.rename(columns=header_mapping)[header_mapping.values()]
    df["component_id"] = f"{component_prefix} " + df.index.astype(str).str.zfill(6)
    df["spec_idx"] = df["spec_idx"].apply(lambda x: [x])
    df["log_spec_idx"] = True

    # OSKAR stores FWHM, the SKA LSM stores semi-major/minor axes
    df["a_arcsec"] = df["a_arcsec"] / 2
    df["b_arcsec"] = df["b_arcsec"] / 2
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
        description="Convert OSKAR CSV format to SKA LSM CSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input OSKAR CSV file.",
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
