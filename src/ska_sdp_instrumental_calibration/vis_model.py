# pylint: disable=too-many-ancestors,too-many-arguments,too-many-locals
# pylint: disable=invalid-name, unexpected-keyword-arg

"""
Visibility data model.
"""
import numpy
import xarray
from astropy.time import Time
from ska_sdp_datamodels.science_data_model import PolarisationFrame


class Visibility(xarray.Dataset):
    """
    Visibility xarray.Dataset class

    This is a substitute for ska_sdp_datamodels.visibility.Visibility, with
    the baselines dimension replaced by 1D integer array, baselineid.
    Arrays baselines, antenna1 and antenna2 are retained as data variables.
    This is to avoid MultiIndex confusion when using dask arrays.

    Note that this is a temporary solution. The intention is to switch to
    xradio data models once they are ready for general use.
    """

    @classmethod
    def constructor(
        cls,
        frequency=None,
        channel_bandwidth=None,
        phasecentre=None,
        configuration=None,
        uvw=None,
        time=None,
        vis=None,
        weight=None,
        integration_time=None,
        flags=None,
        baselines=None,
        polarisation_frame=PolarisationFrame("stokesI"),
        source="anonymous",
        scan_id=0,
        scan_intent="none",
        execblock_id=0,
        meta=None,
        low_precision="float64",
    ):
        """Visibility

        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param uvw: UVW coordinates (m) [:, nant, nant, 3]
        :param time: Time (UTC) [:]
        :param baselines: List of baselines
        :param flags: Flags [:, nant, nant, nchan]
        :param weight: [:, nant, nant, nchan, npol]
        :param integration_time: Integration time [:]
        :param polarisation_frame: Polarisation_Frame
                e.g. Polarisation_Frame("linear")
        :param source: Source name
        :param scan_id: Scan number ID (integer)
        :param scan_intent: Intent for the scan (string)
        :param execblock_id: Execution block ID (integer)
        :param meta: Meta info
        """
        if weight is None:
            weight = numpy.ones(vis.shape)
        else:
            assert weight.shape == vis.shape

        if integration_time is None:
            integration_time = numpy.ones_like(time)
        else:
            assert len(integration_time) == len(time)

        baselineid = numpy.arange(len(baselines))

        # Define the names of the dimensions
        coords = {  # pylint: disable=duplicate-code
            "time": time,
            "baselineid": baselineid,
            "frequency": frequency,
            "polarisation": polarisation_frame.names,
            "spatial": ["u", "v", "w"],
        }

        datavars = {}
        datavars["integration_time"] = xarray.DataArray(
            integration_time.astype(low_precision),
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["datetime"] = xarray.DataArray(
            Time(time / 86400.0, format="mjd", scale="utc").datetime64,
            dims=["time"],
            attrs={"units": "s"},
        )
        datavars["vis"] = xarray.DataArray(
            vis,
            dims=["time", "baselineid", "frequency", "polarisation"],
            attrs={"units": "Jy"},
        )
        datavars["weight"] = xarray.DataArray(
            weight.astype(low_precision),
            dims=["time", "baselineid", "frequency", "polarisation"],
        )
        datavars["flags"] = xarray.DataArray(
            flags.astype(int),
            dims=["time", "baselineid", "frequency", "polarisation"],
        )
        datavars["uvw"] = xarray.DataArray(
            uvw,
            dims=["time", "baselineid", "spatial"],
            attrs={"units": "m"},
        )

        datavars["channel_bandwidth"] = xarray.DataArray(
            channel_bandwidth, dims=["frequency"], attrs={"units": "Hz"}
        )

        # Move the baselines dimension to data variables.
        #  - TODO reduce duplication of indices.
        datavars["baselines"] = xarray.DataArray(
            baselines.baselines.data, dims=["baselineid"]
        )
        datavars["antenna1"] = xarray.DataArray(
            baselines.antenna1.data, dims=["baselineid"]
        )
        datavars["antenna2"] = xarray.DataArray(
            baselines.antenna2.data, dims=["baselineid"]
        )

        datavars["imaging_weight"] = xarray.DataArray(
            weight.astype(low_precision),
            dims=["time", "baselineid", "frequency", "polarisation"],
        )

        attrs = {}
        attrs["data_model"] = "Visibility"
        attrs["configuration"] = configuration  # Antenna/station configuration
        attrs["source"] = source
        attrs["phasecentre"] = phasecentre
        attrs["_polarisation_frame"] = polarisation_frame.type
        attrs["scan_id"] = scan_id
        attrs["scan_intent"] = scan_intent
        attrs["execblock_id"] = execblock_id
        attrs["meta"] = meta

        return cls(datavars, coords=coords, attrs=attrs)

    def __sizeof__(self):
        """Override default method to return size of dataset
        :return: int
        """
        # Dask uses sizeof() class to get memory occupied by various data
        # objects. For custom data objects like this one, dask falls back to
        # sys.getsizeof() function to get memory usage. sys.getsizeof() in
        # turns calls __sizeof__() magic method to get memory size. Here we
        # override the default method (which gives size of reference table)
        # to return size of Dataset.
        return int(self.nbytes)

    def copy(self, deep=False, data=None, zero=False):
        """
        Copy Visibility

        :param deep: perform deep-copy
        :param data: data to use in new object; see docstring of
                     xarray.core.dataset.Dataset.copy
        :param zero: if True, set visibility data to zero in copied object
        """
        new_vis = super().copy(deep=deep, data=data)
        if zero:
            new_vis["vis"].data[...] = 0.0

        return new_vis
