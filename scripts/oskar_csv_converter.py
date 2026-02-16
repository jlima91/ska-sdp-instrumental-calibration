#!/usr/bin/env python
"""
Convert OSKAR CSV format to new sky model CSV format.

Old OSKAR format columns:
RA (deg), Dec (deg), I (Jy), Q (Jy), U (Jy), V (Jy), Ref. freq. (Hz),
Spectral index, Rotation measure (rad/m^2), FWHM major (arcsec),
FWHM minor (arcsec), Position angle (deg)

New format columns:
component_id, ra, dec, i_pol, major_ax, minor_ax, pos_ang, ref_freq,
spec_idx, log_spec_idx

Usage:
    python scripts/oskar_csv_converter.py input.csv
    python scripts/oskar_csv_converter.py input.csv output.csv
    python scripts/oskar_csv_converter.py input.csv output.csv --prefix GLEAM
"""

import argparse

import pandas as pd

from ska_sdp_instrumental_calibration.data_managers.sky_model.sky_model_reader import (
    SKY_MODEL_CSV_HEADER,
    ComponentConverters,
)
from ska_sdp_instrumental_calibration.data_managers.sky_model.utils import (
    write_csv,
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


def convert_oskar_to_new_format(
    input_file: str,
    output_file: str,
    component_prefix: str,
) -> pd.DataFrame:
    """
    Convert an OSKAR format CSV file to the new sky model format.

    Parameters
    ----------
    input_file : str
        Path to the input OSKAR CSV file.
    output_file : str, optional
        Path for the output CSV file in new format. Default is
         "converted_sky_model.csv".
    component_prefix : str, optional
        Prefix for generating component IDs. Default is "Component".
    log_spec_idx : bool, optional
        Whether the spectral index uses logarithmic model. Default is True.

    Returns
    -------
    pd.DataFrame
        The converted DataFrame.
    """
    # Read the old file
    df = pd.read_csv(
        input_file,
        sep=",",
        comment="#",
        names=OSKAR_HEADERS,
        dtype=float,
    )

    new_df = pd.DataFrame()

    new_df["component_id"] = [
        f"{component_prefix} {i:06d}" for i in range(len(df))
    ]

    # Map columns from old to new format
    new_df["ra"] = df["RA (deg)"]
    new_df["dec"] = df["Dec (deg)"]
    new_df["i_pol"] = df["I (Jy)"]
    new_df["major_ax"] = df["FWHM major (arcsec)"]
    new_df["minor_ax"] = df["FWHM minor (arcsec)"]
    new_df["pos_ang"] = df["Position angle (deg)"]
    new_df["ref_freq"] = df["Ref. freq. (Hz)"]

    new_df["spec_idx"] = df["Spectral index"].apply(
        lambda x: [x] if pd.notna(x) else [0.0]
    )

    new_df["log_spec_idx"] = True

    # Convert to components and write to CSV
    components = ComponentConverters.df_to_components(new_df)
    rows = [["#" + ",".join(SKY_MODEL_CSV_HEADER)]]

    rows.extend(
        [ComponentConverters.to_csv_row(component) for component in components]
    )

    write_csv(output_file, rows)
    print(f"Converted {len(new_df)} components from {input_file} to {output_file}")
    return new_df


def main():
    parser = argparse.ArgumentParser(
        description="Convert OSKAR CSV format to new sky model CSV format.",
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
        default="converted_sky_model.csv",
        help="Path for the output CSV file in new format.",
    )
    parser.add_argument(
        "--comp-prefix",
        type=str,
        default="Component",
        help="Prefix for generating component IDs. Default: GLEAM",
    )
   

    args = parser.parse_args()

    convert_oskar_to_new_format(
        input_file=args.input_csv,
        output_file=args.output_csv,
        component_prefix=args.comp_prefix,
    )


if __name__ == "__main__":
    main()