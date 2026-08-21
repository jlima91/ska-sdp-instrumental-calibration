import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

resource_root_path = script_path = Path(__file__).resolve().parent

MS_TAR = Path(f"{resource_root_path}/test.ms.tgz")
LSM_CSV = f"{resource_root_path}/sky_model.csv"
CONFIG_PATH = f"{resource_root_path}/config.yml"


@dataclass
class TestResource:
    ms_files: list[str]
    config: str
    lsm_csv: str


def init_data(temp_dir: Path):
    with tarfile.open(MS_TAR, "r:*") as tar:
        tar.extractall(path=temp_dir)
        ms_name = tar.getnames()[0].split("/")[0]
        ms_path = (temp_dir / Path(ms_name)).as_posix()
        ms_path_2 = (temp_dir / Path(f"a_{ms_name}")).as_posix()
        shutil.copytree(ms_path, ms_path_2)

    return TestResource([ms_path, ms_path_2], CONFIG_PATH, LSM_CSV)
