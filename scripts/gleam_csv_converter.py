#!/usr/bin/env python
"""
Convert the GLEAM-EGC catalogue (gleamegc.dat) to SKA LSM CSV format.

The GLEAM-EGC catalogue is a fixed width text file, with one source per
line. The following columns are used:

GLEAM name, RA (deg), Dec (deg), Fintwide (Jy), a_wide (arcsec),
b_wide (arcsec), pa_wide (deg), psf_a_wide (arcsec), psf_b_wide (arcsec),
psf_pa_wide (deg), alpha, Fintfit200 (Jy)

The flux and spectral index are taken from 'Fintfit200' and 'alpha', at a
reference frequency of 200 MHz. If these are unavailable (marked as '---'),
'Fintwide' and a nominal spectral index (--alpha0) are used instead.

The fitted source shape is deconvolved from the synthesised beam (psf).
GLEAM labels the shape columns as semi-major / semi-minor axes, but they
actually hold the full width at half maximum (FWHM), while the SKA LSM
stores semi-axes, so the axes are halved. Both formats measure the position
angle from North through East, so it is unchanged.

The output starts with a CATALOGUE_METADATA_* header, which the pipeline
needs to read the file. The GLEAM-EGC metadata (catalogue name, version,
reference and frequency range) can be overridden with the --catalogue-name,
--catalogue-version, --description and --author options.

SKA LSM format columns:
component_id, source_id, ra_deg, dec_deg, i_pol_jy, ref_freq_hz, epoch,
a_arcsec, b_arcsec, pa_deg, spec_idx, log_spec_idx

For more details on the SKA LSM format, please visit:
https://developer.skao.int/projects/ska-sdp-global-sky-model/en/latest/design/lsm-file-structure.html

Usage:
    python scripts/gleam_csv_converter.py gleamegc.dat
    python scripts/gleam_csv_converter.py gleamegc.dat output.csv
    python scripts/gleam_csv_converter.py gleamegc.dat --flux-limit 0.1
    python scripts/gleam_csv_converter.py gleamegc.dat --alpha0 -0.7
    python scripts/gleam_csv_converter.py gleamegc.dat --comp-prefix GLEAM \
        --author "A. Person"
"""

import argparse
from dataclasses import replace
from typing import Optional

from ska_sdp_datamodels.sky_model import write_skymodel_to_lsm_csv

from ska_sdp_instrumental_calibration.data_managers.sky_model import (
    read_gleamegc,
)


def convert_gleam_to_ska_lsm(
    input_file: str,
    output_file: str,
    flux_limit: float = 0.0,
    alpha0: float = -0.78,
    component_prefix: Optional[str] = None,
    catalogue_name: Optional[str] = None,
    catalogue_version: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None,
):
    """
    Convert a GLEAM-EGC catalogue file to the SKA LSM CSV format.

    Parameters
    ----------
    input_file : str
        Path to the input GLEAM-EGC catalogue file (gleamegc.dat).
    output_file : str
        Path for the output CSV file in SKA LSM format.
    flux_limit : float, default: 0.0
        Only components with a wide-band integrated flux ('Fintwide')
        greater than this limit are kept. Units: [Jy]
    alpha0 : float, default: -0.78
        The nominal spectral index to use when the fitted flux and spectral
        index are unavailable in the catalogue.
    component_prefix : str, optional
        Prefix added to the GLEAM names to form the component IDs
        (e.g. "GLEAM" gives "GLEAM J000000-123456").
    catalogue_name : str, optional
        Overrides the catalogue name in the metadata header.
    catalogue_version : str, optional
        Overrides the catalogue version in the metadata header.
    description : str, optional
        Catalogue description written to the metadata header.
    author : str, optional
        Catalogue author written to the metadata header.
    """
    sky_model = read_gleamegc(input_file, alpha0=alpha0, flux_limit=flux_limit)

    if component_prefix:
        components = [
            replace(
                component,
                component_id=f"{component_prefix} {component.component_id}",
            )
            for component in sky_model.components.values()
        ]
        sky_model.components = {
            component.component_id: component for component in components
        }

    overrides = {
        "catalogue_name": catalogue_name,
        "version": catalogue_version,
        "description": description,
        "author": author,
    }
    sky_model.metadata = replace(
        sky_model.metadata,
        **{key: val for key, val in overrides.items() if val is not None},
    )

    write_skymodel_to_lsm_csv(sky_model, output_file)

    print(
        f"Converted {sky_model.num_components} components from {input_file} "
        f"to {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert the GLEAM-EGC catalogue to SKA LSM CSV format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input GLEAM-EGC catalogue file (gleamegc.dat).",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=str,
        default="./converted_sky_model.csv",
        help="Path for the output CSV file in SKA LSM format.",
    )
    parser.add_argument(
        "--flux-limit",
        type=float,
        default=0.0,
        help="Only keep components with a wide-band integrated flux "
        "greater than this limit (Jy). Default: 0.0",
    )
    parser.add_argument(
        "--alpha0",
        type=float,
        default=-0.78,
        help="Nominal spectral index for components without a fitted "
        "spectral index. Default: -0.78",
    )
    parser.add_argument(
        "--comp-prefix",
        type=str,
        default=None,
        help="Prefix added to the GLEAM names to form the component IDs. "
        "Default: no prefix",
    )
    parser.add_argument(
        "--catalogue-name",
        type=str,
        default=None,
        help="Catalogue name for the metadata header. Default: GLEAM-EGC",
    )
    parser.add_argument(
        "--catalogue-version",
        type=str,
        default=None,
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

    convert_gleam_to_ska_lsm(
        input_file=args.input_file,
        output_file=args.output_csv,
        flux_limit=args.flux_limit,
        alpha0=args.alpha0,
        component_prefix=args.comp_prefix,
        catalogue_name=args.catalogue_name,
        catalogue_version=args.catalogue_version,
        description=args.description,
        author=args.author,
    )


if __name__ == "__main__":
    main()
