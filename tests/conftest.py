"""
test utilities for ska_sdp_instrumental_calibration
"""

import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from ska_sdp_datamodels.calibration.calibration_create import (
    create_gaintable_from_visibility,
)
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility.vis_create import create_visibility
from ska_sdp_datamodels.visibility.vis_io_ms import export_visibility_to_ms

ms_name = "test.ms"


@pytest.fixture
def generate_vis():
    """Fixture to build Visibility and GainTable datasets."""
    # Create the Visibility dataset
    config = create_named_configuration("LOWBD2")
    AA1 = (
        np.concatenate(
            (
                345 + np.arange(6),  # S8-1:6
                351 + np.arange(4),  # S9-1:4
                429 + np.arange(6),  # S10-1:6
                465 + np.arange(4),  # S16-1:4
            )
        )
        - 1
    )
    mask = np.isin(config.id.data, AA1)
    nstations = config.stations.shape[0]
    config = config.sel(indexers={"id": np.arange(nstations)[mask]})
    # Reset relevant station parameters
    nstations = config.stations.shape[0]
    config.stations.data = np.arange(nstations).astype("str")
    config = config.assign_coords(id=np.arange(nstations))
    # config.attrs["name"] = config.name+"-AA1"
    config.attrs["name"] = "AA1-Low"
    vis = create_visibility(
        config=config,
        times=np.arange(3) * 0.9 / 3600 * np.pi / 12,
        frequency=150e6 + 1e6 * np.arange(4),
        channel_bandwidth=[1e6] * 4,
        phasecentre=SkyCoord(ra=0, dec=-27, unit="degree"),
        polarisation_frame=PolarisationFrame("linear"),
        weight=1.0,
    )
    # Put a point source at phase centre
    vis.vis.data[..., :] = [1, 0, 0, 1]

    # Create the GainTable dataset
    jones = create_gaintable_from_visibility(vis, jones_type="B")
    jones.gain.data[..., 0, 0] = 1 - 0.1j
    jones.gain.data[..., 1, 1] = 3 + 0j
    jones.gain.data += np.random.normal(0, 0.2, jones.gain.shape)
    jones.gain.data += np.random.normal(0, 0.2, jones.gain.shape) * 1j

    return vis, jones


@pytest.fixture(scope="module")
def oskar_ms(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extracts a tar archive to the same directory as the archive.

    Returns:
        Path: string name of the extracted directory
    """
    archive_path = Path("data/OSKAR_MOCK.ms.tar.gz")
    datasets_tmpdir = Path(tmp_path_factory.mktemp("pytest_datasets"))
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(datasets_tmpdir)
        return datasets_tmpdir / tar.getnames()[0]


@pytest.fixture
def generate_ms(tmp_path, generate_vis):
    """Create and later delete test MSv2."""
    vis, _ = generate_vis
    ms_path = f"{tmp_path}/{ms_name}"
    export_visibility_to_ms(ms_path, [vis])

    yield ms_path

    shutil.rmtree(ms_path)


@pytest.fixture
def generate_vis_mvis_gain_ndarray_data(generate_vis):
    """Create mock visibility data for testing."""
    vis, gaintable = generate_vis

    vis_vis = vis.vis.values
    vis_flags = vis.flags.values
    vis_weight = vis.weight.values

    ntime, nbaseline, nfreq, npol = vis_vis.shape
    model_vis = np.random.randn(
        ntime, nbaseline, nfreq, npol
    ) + 1j * np.random.randn(ntime, nbaseline, nfreq, npol)
    model_flags = np.zeros((ntime, nbaseline, nfreq, npol), dtype=bool)

    gain_gain = gaintable.gain.values
    gain_weight = gaintable.weight.values
    gain_residual = gaintable.residual.values

    ant1 = vis.antenna1.values
    ant2 = vis.antenna2.values

    return {
        "vis_vis": vis_vis,
        "vis_flags": vis_flags,
        "vis_weight": vis_weight,
        "model_vis": model_vis,
        "model_flags": model_flags,
        "gain_gain": gain_gain,
        "gain_weight": gain_weight,
        "gain_residual": gain_residual,
        "ant1": ant1,
        "ant2": ant2,
        "nchannels": nfreq,
    }
