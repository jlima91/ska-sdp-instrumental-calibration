"""Module to initialise a pipeline from input parameters"""

from astropy.coordinates import SkyCoord

from ska_sdp_instrumental_calibration.logger import setup_logger
from ska_sdp_instrumental_calibration.numpy_processors.lsm import (
    generate_lsm_from_csv,
    generate_lsm_from_gleamegc,
)
from ska_sdp_instrumental_calibration.workflow.utils import create_demo_ms

logger = setup_logger("workflow.pipeline_config")


class PipelineConfig:
    """Class to store pipeline config parameters.

    Attributes:
        config: Dictionary

    Args:
        config (dict):
            Input dictionary of pipeline configuration parameters
        dask_scheduler_address (str, optional)
            Dask cluster IP, (e.g. cluster.scheduler_address). Default is None,
            in which case a dask.distributed.LocalCluster scheduler_address
            will be used.
        end_to_end_subbands (bool):
            If true, vis ingest, prediction and solving will all occur within a
            single end-to-end task for each sub-band. Otherwise separate tasks
            are used for each of the steps. Defaults to True.
        h5parm_name (str):
            Output H5Parm filename. Defaults to "cal_solutions.h5".
        hdf5_name (str):
            Output HDF5 filename. Defaults to None.
        ms_name (str):
            Input MSv2 filename. Defaults to "demo.ms". If the filename is
            "demo.ms", a demo dataset will be generated and written to this
            file. See also parameters ntimes, nchannels, delays, gains,
            leakage and rotation.
        ntimes (int):
            If filename is "demo.ms", this sets the number of simulated time
            steps. Default is 1.
        nchannels (int):
            If filename is "demo.ms", this sets the number of simulated
            frequency channels. Default is 64.
        delays (bool):
            If filename is "demo.ms", this specifies whether or not random
            station delay corrumptions should be included in the simulation.
            Default is False.
        gains (bool):
            If filename is "demo.ms", this specifies whether or not random
            station gain corrumptions should be included in the simulation.
            Default is True.
        leakage (bool):
            If filename is "demo.ms", this specifies whether or not random
            station leakage corrumptions should be included in the simulation.
            Default is False.
        rotation (bool):
            If filename is "demo.ms", this specifies whether or not
            station-dependent Faraday rotation should be included in the
            simulation. Default is False.
        wide_channels (bool):
            If filename is "demo.ms", this specifies whether or not
            use wider 781.25 kHz channels in the simulation. Default is False,
            which sets 5.4 kHz channels.
        fchunk (int):
            Number of frequency channels to use when chunking datasets. Default
            if 16.
        solver (str):
            Calibration solver to use. Must be supported by ska_sdp_func_python
            calibration function solve_gaintable: "gain_substitution",
            "jones_substitution", "normal_equations" or
            "normal_equations_presum".
        lsm (lsm.Component list, optional):
            Optional list of lsm Component objects to use as the local sky
            model. Default is None, in which case the lsm will be generated
            from the catalogue supplied in parameter gleamfile. Also see
            parameters fov and flux_limit.
        fov (float):
            If lsm is None, this specifies the width of the cone used when
            searching for compoents, in units of degrees. Defaults to 10.
        flux_limit (float):
            If lsm is None, this specifies the flux density limit used when
            searching for compoents, in units of Jy. Defaults to 1.
        gleamfile (str):
            Specifies the location of the gleam catalogue file gleamegc.dat.
            If lsm is None, the sky model must be specified by either a
            gleamegc catalogue file or a csv file.
            GLEAMEGC can be downloaded from VizieR:
            "wget https://cdsarc.cds.unistra.fr/ftp/VIII/100/gleamegc.dat.gz"
            This parameter is deprecated and will be replaced with parameters
            related to ska-sdp-global-sky-model.
        csvfile (str):
            Specifies the location of a CSV sky component list.
            If lsm is None, the sky model must be specified by either a
            gleamegc catalogue file or a csv file.
        beam_type (str):
            Type of beam model to use. Should be "everybeam" or "none".
            Defaults to "everybeam".
        eb_coeffs (str):
            If beam_type is "everybeam", this specifies the location of the
            everybeam coeffs directory. For example
            "./ska-sdp-func-everybeam/coeffs".
        eb_ms (str):
            If beam_type is "everybeam" but dataset ms_name does not have all
            of the metadata required by everybeam, this parameter is used to
            specify a separate dataset to use when setting up the beam models.
        normalise_at_beam_centre (bool):
            If true, before running calibration, multiply vis and model vis by
            the inverse of the beam response in the beam pointing direction.
    """

    def __init__(self, config):
        """Parse input parameters."""

        self.config = config

        # Dask info
        self.dask_scheduler_address = config.get(
            "dask_scheduler_address", None
        )

        # Whether or not to combine processing tasks into a single dask task
        self.end_to_end_subbands = config.get("end_to_end_subbands", True)

        # Input MSv2 filename
        self.ms_name = config.get("ms_name", "demo.ms")

        # Output hdf5 files
        self.h5parm_name = config.get("h5parm_name", "cal_solutions.h5")
        self.hdf5_name = config.get("hdf5_name", None)

        # Check whether or not input data need to be simulated
        self.do_simulation = False
        if self.ms_name == "demo.ms":
            logger.info("Simulating demo MSv2 input")
            self.do_simulation = True

        # Simulation parameters. Only used if self.do_simulation is True
        self.ntimes = config.get("ntimes", 1)
        self.nchannels = config.get("nchannels", 64)
        # Simulation station corruptions
        self.delays = config.get("delays", False)
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
        self.fov = config.get("fov", 10)
        self.flux_limit = config.get("flux_limit", 1)
        self.gleamfile = config.get("gleamfile", None)
        self.csvfile = config.get("csvfile", None)
        # Check required external data
        if self.lsm is None:
            if self.gleamfile is None and self.csvfile is None:
                raise ValueError("An LSM or catalogue file is required")
            elif self.gleamfile is not None and self.csvfile is not None:
                raise ValueError("Specify only a single sky model file")
        else:
            if self.gleamfile is not None or self.csvfile is not None:
                raise ValueError("Specify only a LSM or a sky model file")

        # Beam model info
        self.beam_type = config.get("beam_type", "everybeam")
        self.eb_coeffs = config.get("eb_coeffs", None)
        self.eb_ms = config.get("eb_ms", self.ms_name)
        self.norm_beam_centre = config.get("normalise_at_beam_centre", False)
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
            lsm = []
            if self.lsm is None:
                # Get the LSM (single call for all channels / dask tasks)
                logger.info("Generating LSM for simulation with:")
                logger.info(f" - Search radius: {self.fov/2} deg")
                logger.info(f" - Flux limit: {self.flux_limit} Jy")
                if self.gleamfile is not None:
                    logger.info(f" - GLEAMEGC file: {self.gleamfile}")
                    lsm = generate_lsm_from_gleamegc(
                        gleamfile=self.gleamfile,
                        phasecentre=phasecentre,
                        fov=self.fov,
                        flux_limit=self.flux_limit,
                    )
                elif self.csvfile is not None:
                    logger.info(f" - csv file: {self.csvfile}")
                    lsm = generate_lsm_from_csv(
                        csvfile=self.csvfile,
                        phasecentre=phasecentre,
                        fov=self.fov,
                        flux_limit=self.flux_limit,
                    )
                else:
                    raise ValueError("Unknown sky model")
            else:
                lsm = self.lsm
            logger.info(f"LSM contains {len(lsm)} components")

            return create_demo_ms(
                ms_name=self.ms_name,
                ntimes=self.ntimes,
                nchannels=self.nchannels,
                delays=self.delays,
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
