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

SPLINE_DATA_PATH = __resource_dir__ / "SKA_Low_AA2_SP5175_spline_data.npz"
