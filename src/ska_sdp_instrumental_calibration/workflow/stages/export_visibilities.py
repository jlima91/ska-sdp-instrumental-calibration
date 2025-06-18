from ska_sdp_piper.piper.configurations import ConfigParam, Configuration
from ska_sdp_piper.piper.stage import ConfigurableStage

from ska_sdp_instrumental_calibration.data_managers.dask_wrappers import (
    apply_gaintable_to_dataset,
)


@ConfigurableStage(
    "export_visibilities",
    configuration=Configuration(
        apply_gaintable=ConfigParam(
            str,
            default=None,
            description="Apply gaintable to vis",
            allowed_values=["vis"],
            nullable=True,
        ),
    ),
)
def export_visibilities_stage(upstream_output, apply_gaintable):
    """
    Apply gaintable and export visibilities.

    Parameters
    -----------
        upstream_output: dict
            Output from upstream stage.
        apply_gaintable: str
            Apply gaintable to vis and/or modelvis.

    Returns
    --------
        dict
            Upstream output with corrected vis and/or modelvis.
    """
    gaintable = upstream_output["gaintable"]
    vis = upstream_output["vis"]

    if apply_gaintable == "vis":
        corrected_vis = apply_gaintable_to_dataset(vis, gaintable)
        upstream_output["corrected_vis"] = corrected_vis

    return upstream_output
