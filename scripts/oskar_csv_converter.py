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
    ComponentConverters,
    export_lsm_to_csv,
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
):
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

    # Headers after I are optional
    # If those columns are None, fill in 0 instead
    for header in OSKAR_HEADERS[3:]:
        df[header] = df[header].fillna(0)

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
    df["spec_idx"] = df["spec_idx"].apply(lambda x: [x] if pd.notna(x) else [0.0])
    df["log_spec_idx"] = True

    # New column values are of 'semi'-major/minor axis
    df["a_arcsec"] = df["a_arcsec"] / 2
    df["b_arcsec"] = df["b_arcsec"] / 2
    df["source_id"] = ""
    df["epoch"] = 0.0
    # Convert to components and write to CSV
    components = ComponentConverters.df_to_components(df)
    export_lsm_to_csv(components, output_file)

    print(f"Converted {len(df)} components from {input_file} to {output_file}")


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
        default="./converted_sky_model.csv",
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
