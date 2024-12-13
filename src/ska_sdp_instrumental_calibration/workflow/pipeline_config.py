"""Module to initialise a pipeline from input parameters"""

from astropy.coordinates import SkyCoord

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.processing_tasks.lsm import generate_lsm
from ska_sdp_instrumental_calibration.workflow.utils import create_demo_ms

logger = setup_logger("workflow.pipeline_config")


class PipelineConfig:
    """Class to store pipeline config parameters.

    Attributes:
        config: Dictionary
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Input dictionary of pipeline configuration parameters
        dask_cluster : Dask cluster, optional
            Dask cluster object containing scheduler and workers. Such as a
            dask_jobqueue.SLURMCluster. Default is None, in which case a
            dask.distributed.LocalCluster will be used.
        hdf5_name : str
            Output hdf5 filename. Defaults to "demo.hdf5".
        ms_name : str
            Input MSv2 filename. Defaults to "demo.ms". If the filename is
            "demo.ms", a demo dataset will be generated and written to this
            file. See also parameters ntimes, nchannels, gains, leakage and
            rotation.
        ntimes : int
            If filename is "demo.ms", this sets the number of simulated time
            steps. Default is 1.
        nchannels : int
            If filename is "demo.ms", this sets the number of simulated
            frequency channels. Default is 64.
        gains : bool
            If filename is "demo.ms", this specifies whether or not random
            station gain corrumptions should be included in the simulation.
            Default is True.
        leakage : bool
            If filename is "demo.ms", this specifies whether or not random
            station leakage corrumptions should be included in the simulation.
            Default is False.
        rotation : bool
            If filename is "demo.ms", this specifies whether or not
            station-dependent Faraday rotation should be included in the
            simulation. Default is False.
        wide_channels : bool
            If filename is "demo.ms", this specifies whether or not
            use wider 781.25 kHz channels in the simulation. Default is False,
            which sets 5.4 kHz channels.
        fchunk : int
            Number of frequency channels to use when chunking datasets. Default
            if 16.
        solver : str
            Calibration solver to use. Must be supported by ska_sdp_func_python
            calibration function solve_gaintable: "gain_substitution",
            "jones_substitution", "normal_equations" or
            "normal_equations_presum".
        lsm : lsm.Component list, optional
            Optional list of lsm Component objects to use as the local sky
            model. Default is None, in which case the lsm will be generated
            from the catalogue supplied in parameter gleamfile. Also see
            parameters fov and flux_limit.
        fov : float
            If lsm is None, this specifies the width of the cone used when
            searching for compoents, in units of degrees. Defaults to 10.
        flux_limit : float
            If lsm is None, this specifies the flux density limit used when
            searching for compoents, in units of Jy. Defaults to 1.
        gleamfile : str
            If lsm is None, this specifies the location of gleam catalogue
            file gleamegc.dat. This can be downloaded from VizieR:
            "wget https://cdsarc.cds.unistra.fr/ftp/VIII/100/gleamegc.dat.gz"
            This parameter is deprecated and will be replaced with parameters
            related to ska-sdp-global-sky-model.
        beam_type : str
            Type of beam model to use. Should be "everybeam" or "none".
            Defaults to "everybeam".
        eb_coeffs : str
            If beam_type is "everybeam", this specifies the location of the
            everybeam coeffs directory. For example
            "./ska-sdp-func-everybeam/coeffs".
        eb_ms: str
            If beam_type is "everybeam" but dataset ms_name does not have all
            of the metadata required by everybeam, this parameter is used to
            specify a separate dataset to use when setting up the beam models.
        """

        self.config = config

        # Dask info
        self.dask_cluster = config.get("dask_cluster", None)

        # Output hdf5 filename
        self.hdf5_name = config.get("hdf5_name", "demo.hdf5")

        # Input MSv2 filename
        self.ms_name = config.get("ms_name", "demo.ms")

        # Check whether or not input data need to be simulated
        self.do_simulation = False
        if self.ms_name == "demo.ms":
            logger.info("Simulating demo MSv2 input")
            self.do_simulation = True

        # Simulation parameters. Only used if self.do_simulation is True
        self.ntimes = config.get("ntimes", 1)
        self.nchannels = config.get("nchannels", 64)
        # Simulation station corruptions
        self.gains = config.get("gains", True)
        self.leakage = config.get("leakage", False)
        self.rotation = config.get("rotation", False)
        self.wide_channels = config.get("wide_channels", False)

        # The number of channels per frequency chunk
        self.fchunk = config.get("fchunk", 16)

        # Solver to use
        self.solver = config.get("solver", "gain_substitution")
        # Should be supported by ska_sdp_func_python solve_gaintable
        # Could just leave this check for solve_gaintable to deal with...
        # But could also use other solver functions, like ska_sdp_func_python
        # solve_ionosphere, so coordinate that here.
        if self.solver not in [
            "gain_substitution",
            "jones_substitution",
            "normal_equations",
            "normal_equations_presum",
        ]:
            raise ValueError(f"Unknown calibration solver: {self.solver}")

        # Sky model info
        self.lsm = config.get("lsm", None)
        self.fov = config.get("fov_deg", 10)
        self.flux_limit = config.get("flux_limit", 1)
        self.gleamfile = config.get("gleamfile", None)
        # Check required external data
        if self.lsm is None and self.gleamfile is None:
            raise ValueError("Either a LSM or a catalogue file is required")

        # Beam model info
        self.beam_type = config.get("beam_type", "everybeam")
        self.eb_coeffs = config.get("eb_coeffs", None)
        self.eb_ms = config.get("eb_ms", self.ms_name)
        if self.beam_type.lower() == "everybeam":
            # Required external data
            if self.eb_coeffs is None:
                raise ValueError(
                    "Path to Everybeam coeffs directory is required"
                )
            logger.info(
                f"Initialising the EveryBeam telescope model with {self.eb_ms}"
            )
        elif self.beam_type.lower() == "none":
            logger.info("Predicting visibilities without a beam")
        else:
            raise ValueError(f"Unknown beam type: {self.beam_type}")

    def simulate_input_dataset(self):
        """
        Simulate a small MSv2 dataset for demo and testing purposes.

        :return: GainTable applied to data
        """
        if self.do_simulation:
            # Generate a demo MSv2 Measurement Set
            logger.info(
                f"Generating a demo MSv2 Measurement Set {self.ms_name}"
            )
            phasecentre = SkyCoord(ra=0.0, dec=-27.0, unit="degree")
            if self.lsm is None:
                # Get the LSM (single call for all channels / dask tasks)
                logger.info("Generating LSM for simulation with:")
                logger.info(f" - Catalogue file: {self.gleamfile}")
                logger.info(f" - Search radius: {self.fov/2} deg")
                logger.info(f" - Flux limit: {self.flux_limit} Jy")
                lsm = generate_lsm(
                    gleamfile=self.gleamfile,
                    phasecentre=phasecentre,
                    fov=self.fov,
                    flux_limit=self.flux_limit,
                )
                logger.info(f"LSM: found {len(lsm)} components")
            else:
                lsm = self.lsm

            return create_demo_ms(
                ms_name=self.ms_name,
                ntimes=self.ntimes,
                nchannels=self.nchannels,
                gains=self.gains,
                leakage=self.leakage,
                rotation=self.rotation,
                wide_channels=self.wide_channels,
                phasecentre=phasecentre,
                lsm=lsm,
                beam_type=self.beam_type,
                eb_coeffs=self.eb_coeffs,
                eb_ms=self.eb_ms,
            )
        else:
            return None
