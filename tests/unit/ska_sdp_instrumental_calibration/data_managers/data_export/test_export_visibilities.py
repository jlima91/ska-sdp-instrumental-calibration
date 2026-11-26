import dask.array as da
import numpy as np
import pytest
import xarray as xr
from casacore.tables import table

from ska_sdp_instrumental_calibration.data_managers.data_export.export_visibilities import (  # noqa: E501
    export_msv4_to_ps,
    msv4_to_visibility,
)
from ska_sdp_instrumental_calibration.data_managers.processing_set import (
    MJD_UNIX_OFFSET_SECONDS,
)
from ska_sdp_instrumental_calibration.stages.export_visibilities import (
    _export_visibility_to_ms,
)

# antenna_xds indices of the baselines of the conftest MSv4 node, in the
# antenna order of the data: the second baseline has antenna1 > antenna2.
ANTENNA1 = [0, 2, 0]
ANTENNA2 = [1, 1, 2]


class TestMSv4ToVisibility:
    def test_should_keep_data_lazy_and_in_msv4_dtypes(self, make_ms_xdt):
        ms_xdt = make_ms_xdt()

        vis = msv4_to_visibility(ms_xdt)

        for name, msv4_name in [
            ("vis", "VISIBILITY"),
            ("flags", "FLAG"),
            ("weight", "WEIGHT"),
        ]:
            assert isinstance(vis[name].data, da.Array)
            assert vis[name].dtype == ms_xdt.ds[msv4_name].dtype
            assert vis[name].chunks == ms_xdt.ds[msv4_name].chunks
            np.testing.assert_array_equal(
                vis[name].values, ms_xdt.ds[msv4_name].values
            )
        assert isinstance(vis.uvw.data, da.Array)
        assert "time_centroid" not in vis

    def test_should_keep_baselines_in_msv4_antenna_order(self, make_ms_xdt):
        vis = msv4_to_visibility(make_ms_xdt())

        np.testing.assert_array_equal(vis.antenna1.values, ANTENNA1)
        np.testing.assert_array_equal(vis.antenna2.values, ANTENNA2)

    def test_should_negate_uvw_as_the_msv2_writer_negates_it(
        self, make_ms_xdt
    ):
        ms_xdt = make_ms_xdt()

        vis = msv4_to_visibility(ms_xdt)

        np.testing.assert_array_equal(vis.uvw.values, -ms_xdt.ds.UVW.values)

    def test_should_convert_metadata(self, make_ms_xdt):
        ms_xdt = make_ms_xdt()

        vis = msv4_to_visibility(ms_xdt)

        np.testing.assert_array_equal(
            vis.time.values, ms_xdt.ds.time.values + MJD_UNIX_OFFSET_SECONDS
        )
        ntimes = ms_xdt.ds.time.size
        integration_time = ms_xdt.ds.time.attrs["integration_time"]["data"]
        channel_width = ms_xdt.ds.frequency.attrs["channel_width"]["data"]
        antenna_xds = ms_xdt["antenna_xds"].ds
        np.testing.assert_array_equal(
            vis.integration_time.values, np.full(ntimes, integration_time)
        )
        np.testing.assert_array_equal(
            vis.frequency.values, ms_xdt.ds.frequency.values
        )
        np.testing.assert_array_equal(
            vis.channel_bandwidth.values,
            np.full(ms_xdt.ds.frequency.size, channel_width),
        )
        assert vis.visibility_acc.polarisation_frame.type == "linear"
        # MSv2 name, without the "_<FIELD_ID>" suffix added by xradio
        assert vis.source == "field"
        np.testing.assert_allclose(
            [vis.phasecentre.ra.rad, vis.phasecentre.dec.rad],
            ms_xdt[
                "field_and_source_base_xds"
            ].ds.FIELD_PHASE_CENTER_DIRECTION.values[0],
        )
        np.testing.assert_array_equal(
            vis.configuration.names.values, antenna_xds.antenna_name.values
        )
        np.testing.assert_array_equal(
            vis.configuration.xyz.values, antenna_xds.ANTENNA_POSITION.values
        )
        np.testing.assert_allclose(
            [c.to_value("m") for c in vis.configuration.location.geocentric],
            antenna_xds.ANTENNA_POSITION.values[0],
        )

    def test_should_reject_polarizations_unsupported_by_msv2_export(
        self, make_ms_xdt
    ):
        with pytest.raises(ValueError, match="Cannot export polarizations"):
            msv4_to_visibility(make_ms_xdt(polarization=("XX", "XY")))

    def test_should_require_the_integration_time(self, make_ms_xdt):
        ms_xdt = make_ms_xdt()
        ms_xdt.ds.time.attrs.pop("integration_time")

        with pytest.raises(ValueError, match="integration_time"):
            msv4_to_visibility(ms_xdt)


def test_should_write_msv2_rows_as_stored_in_msv4(make_ms_xdt, tmp_path):
    ms_xdt = make_ms_xdt()
    ms_path = (tmp_path / "out.ms").as_posix()

    # The pipeline task manager computes the Visibility before writing
    _export_visibility_to_ms(ms_path, [msv4_to_visibility(ms_xdt).compute()])

    with table(ms_path, ack=False) as ms:
        antenna1 = ms.getcol("ANTENNA1")
        antenna2 = ms.getcol("ANTENNA2")
        time = ms.getcol("TIME")
        uvw = ms.getcol("UVW")
        data = ms.getcol("DATA")
        flags = ms.getcol("FLAG")
    ntimes, nbaselines = ms_xdt.ds.time.size, ms_xdt.ds.baseline_id.size
    # The MSv2 writer sorts the rows of each time by (antenna1, antenna2),
    # keeping the antenna order of each baseline: (0, 1), (0, 2), (2, 1)
    order = [0, 2, 1]

    def rows(msv4_values):
        return msv4_values[:, order].reshape(-1, *msv4_values.shape[2:])

    np.testing.assert_array_equal(
        antenna1, np.tile(np.array(ANTENNA1)[order], ntimes)
    )
    np.testing.assert_array_equal(
        antenna2, np.tile(np.array(ANTENNA2)[order], ntimes)
    )
    np.testing.assert_array_equal(
        time,
        np.repeat(ms_xdt.ds.time.values + MJD_UNIX_OFFSET_SECONDS, nbaselines),
    )
    np.testing.assert_array_equal(uvw, rows(ms_xdt.ds.UVW.values))
    np.testing.assert_array_equal(data, rows(ms_xdt.ds.VISIBILITY.values))
    np.testing.assert_array_equal(flags, rows(ms_xdt.ds.FLAG.values))


class TestExportMSv4ToPS:
    def test_should_write_data_only_when_computed(self, make_ms_xdt, tmp_path):
        ps_path = (tmp_path / "out.ps.zarr").as_posix()

        write = export_msv4_to_ps(ps_path, make_ms_xdt(), "node")

        assert isinstance(write, da.Array)
        written = xr.open_datatree(ps_path, engine="zarr")
        # Only the metadata is written. Unwritten chunks read as fill value.
        assert not written["node"].ds.VISIBILITY.values.any()

    def test_should_write_processing_set_with_node_and_children(
        self, make_ms_xdt, tmp_path
    ):
        ms_xdt = make_ms_xdt()
        ps_path = (tmp_path / "out.ps.zarr").as_posix()

        export_msv4_to_ps(ps_path, ms_xdt, "node").compute()

        written = xr.open_datatree(ps_path, engine="zarr")
        assert written.attrs["type"] == "processing_set"
        assert list(written.children) == ["node"]
        node = written["node"]
        assert set(node.children) == set(ms_xdt.children)
        assert node.attrs["data_groups"] == ms_xdt.attrs["data_groups"]
        xr.testing.assert_identical(
            node.to_dataset(inherit=False), ms_xdt.to_dataset()
        )
        xr.testing.assert_identical(
            node["antenna_xds"].to_dataset(),
            ms_xdt["antenna_xds"].to_dataset(),
        )
        # zarr chunks follow the dask chunks
        assert (
            node.ds.VISIBILITY.encoding["chunks"]
            == ms_xdt.ds.VISIBILITY.data.chunksize
        )

    def test_should_merge_irregular_chunks(self, make_ms_xdt, tmp_path):
        ms_xdt = make_ms_xdt()
        # e.g. model visibilities, chunked per solution interval
        ms_xdt.ds = ms_xdt.to_dataset().chunk(time=(1, 3))
        ps_path = (tmp_path / "out.ps.zarr").as_posix()

        export_msv4_to_ps(ps_path, ms_xdt, "node").compute()

        node = xr.open_datatree(ps_path, engine="zarr")["node"]
        assert node.ds.VISIBILITY.encoding["chunks"][0] == ms_xdt.ds.time.size
        np.testing.assert_array_equal(
            node.ds.VISIBILITY.values, ms_xdt.ds.VISIBILITY.values
        )

    def test_should_ignore_storage_chunks_of_the_source_store(
        self, make_ms_xdt, tmp_path
    ):
        ms_xdt = make_ms_xdt()
        # Incompatible with the dask chunks of the data
        ms_xdt.ds.VISIBILITY.encoding.update(
            chunks=(3, 3, 5, 4), preferred_chunks={"time": 3}
        )
        ps_path = (tmp_path / "out.ps.zarr").as_posix()

        export_msv4_to_ps(ps_path, ms_xdt, "node").compute()

        node = xr.open_datatree(ps_path, engine="zarr")["node"]
        np.testing.assert_array_equal(
            node.ds.VISIBILITY.values, ms_xdt.ds.VISIBILITY.values
        )
        # The source node is unchanged
        assert ms_xdt.ds.VISIBILITY.encoding["chunks"] == (3, 3, 5, 4)
