#!/usr/bin/env python
"""
Convert an SKA LSM CSV file to a sourcedb (makesourcedb) sky model file,
as read by DP3 (e.g. ``gaincal.sourcedb`` or ``predict.sourcedb``).

SKA LSM format columns:
component_id, source_id, ra_deg, dec_deg, i_pol_jy, ref_freq_hz, epoch,
a_arcsec, b_arcsec, pa_deg, spec_idx, log_spec_idx

sourcedb format columns:
Name, Type, Ra, Dec, I, ReferenceFrequency, SpectralIndex, LogarithmicSI,
MajorAxis, MinorAxis, Orientation

The SKA LSM stores the semi-major / semi-minor axes (half of the full width
at half maximum) of Gaussian sources, while sourcedb stores the FWHM, so the
axes are doubled. Both formats measure the position angle from North through
East, so it is unchanged. Components without a shape, or with both axes 0,
are written as point sources.

source_id and epoch are not part of the sourcedb format and are dropped.

DP3 only reads a sky model file directly (without makesourcedb) when its
path ends with ".skymodel" or ".txt".

For more details on the SKA LSM format, please visit:
https://developer.skao.int/projects/ska-sdp-global-sky-model/en/latest/design/lsm-file-structure.html

For more details on the sourcedb format, please visit:
https://www.astron.nl/lofarwiki/doku.php?id=public:user_software:documentation:makesourcedb

Usage:
    python scripts/convert/ska_lsm_to_sourcedb.py sky_model.csv
    python scripts/convert/ska_lsm_to_sourcedb.py sky.csv sky.skymodel
"""

import argparse
import math
from typing import Optional

from ska_sdp_datamodels.sky_model import (
    SkyComponent,
    read_skymodel_from_lsm_csv,
)

SOURCEDB_FORMAT = (
    "FORMAT = Name, Type, Ra, Dec, I, ReferenceFrequency, SpectralIndex, "
    "LogarithmicSI, MajorAxis, MinorAxis, Orientation"
)


def _value_or_none(value: Optional[float]) -> Optional[float]:
    """
    Return None for missing values, which are None or NaN depending on how
    the SkyComponent was created.
    """
    if value is None or math.isnan(value):
        return None
    return value


def _fmt(value: float) -> str:
    """Format a number with full precision, without numpy type names."""
    return repr(float(value))


def component_to_sourcedb_row(component: SkyComponent) -> str:
    """
    Convert a sky component to a row of a sourcedb sky model file.

    Parameters
    ----------
    component : SkyComponent
        Sky component in the SKA LSM convention.

    Returns
    -------
    str
        Row matching ``SOURCEDB_FORMAT``.
    """
    # The LSM pads the spectral index to a fixed length with missing values
    spec_idx = [_value_or_none(coeff) for coeff in component.spec_idx]
    while spec_idx and spec_idx[-1] is None:
        spec_idx.pop()
    spec_idx = [0.0 if coeff is None else coeff for coeff in spec_idx]

    log_spec_idx = (
        True if component.log_spec_idx is None else component.log_spec_idx
    )

    a_arcsec = _value_or_none(component.a_arcsec)
    b_arcsec = _value_or_none(component.b_arcsec)
    if (
        a_arcsec is None
        or b_arcsec is None
        or (a_arcsec == 0 and b_arcsec == 0)
    ):
        source_type = "POINT"
        shape = ["", "", ""]
    else:
        source_type = "GAUSSIAN"
        pa_deg = _value_or_none(component.pa_deg) or 0.0
        # SKA LSM stores semi-axes, sourcedb stores FWHM
        shape = [_fmt(2 * a_arcsec), _fmt(2 * b_arcsec), _fmt(pa_deg)]

    return ",".join(
        [
            component.component_id.replace(" ", "_"),
            source_type,
            f"{_fmt(component.ra_deg)}deg",
            f"{_fmt(component.dec_deg)}deg",
            _fmt(component.i_pol_jy),
            _fmt(component.ref_freq_hz),
            "[" + ",".join(_fmt(coeff) for coeff in spec_idx) + "]",
            "true" if log_spec_idx else "false",
            *shape,
        ]
    )


def convert_ska_lsm_to_sourcedb(input_file: str, output_file: str):
    """
    Convert an SKA LSM CSV file to a sourcedb sky model file.

    Parameters
    ----------
    input_file : str
        Path to the input SKA LSM CSV file.
    output_file : str
        Path for the output sourcedb sky model file.
    """
    sky_model = read_skymodel_from_lsm_csv(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(SOURCEDB_FORMAT + "\n")
        for component in sky_model.components.values():
            f.write(component_to_sourcedb_row(component) + "\n")

    print(
        f"Converted {sky_model.num_components} components from {input_file} "
        f"to {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert SKA LSM CSV format to sourcedb sky model format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input SKA LSM CSV file.",
    )
    parser.add_argument(
        "output_skymodel",
        nargs="?",
        type=str,
        default="./converted_sky_model.skymodel",
        help="Path for the output sourcedb sky model file. DP3 reads it "
        "directly only if it ends with .skymodel or .txt.",
    )

    args = parser.parse_args()

    convert_ska_lsm_to_sourcedb(
        input_file=args.input_csv,
        output_file=args.output_skymodel,
    )


if __name__ == "__main__":
    main()
