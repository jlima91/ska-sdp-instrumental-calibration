from pathlib import Path

__resource_dir__ = Path(__file__).resolve().parent

TEL_MODEL = __resource_dir__ / "SKA-Low_AA2_18S_rigid-rotation_model.tm"
SKY_MODEL = str(__resource_dir__ / "sky_model.csv")
INST_CAL_CONFIG = __resource_dir__ / "inst_cal.yaml"
INST_TARGET_CONFIG = __resource_dir__ / "inst_target_complex_gain.yaml"
CABLE_DELAYS = __resource_dir__ / "cable_length_error_18s.txt"
H5PARM_CONVERTER_SCRIPT = (
    __resource_dir__.parents[2]
    / "scripts/ska_low_sim/utils/h5parm_from_oskar_gains.py"
)


TRANSIT_TIME = "2000-01-03 22:33:30.000"

N_STATIONS = 18
START_FREQ_HZ = 50.0e6
END_FREQ_HZ = 125.0e6
CHANNEL_WIDTH_HZ = 75.0e4
SAMPLING_TIME_SEC = 3.0
OBSERVING_TIME_MINS = 1

OUTLIER_STATION_INDICES = [2, 14]
OUTLIER_CHANNEL_INDICES = list(range(50, 60))
OUTLIER_AMPLITUDE = 3.5
OUTLIER_PHASE_DEG = 45.0

EOR2_CAL_RA = 197.914612
EOR2_CAL_DEC = -22.277973

EOR2_TARGET_RA = 129.3395
EOR2_TARGET_DEC = -32.8914
