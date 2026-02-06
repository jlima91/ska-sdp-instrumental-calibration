import tarfile
from dataclasses import dataclass
from pathlib import Path

resource_root_path = script_path = Path(__file__).resolve().parent

MS_TAR = Path(f"{resource_root_path}/demo.ms.tgz")
GLEAMDATA = f"{resource_root_path}/gleamdata.dat"
CONFIG_PATH = f"{resource_root_path}/config.yml"
OSKAR_MOCK_TAR = Path(f"{resource_root_path}/OSKAR_MOCK.ms.tar.gz")


@dataclass
class TestResource:
    ms_file: str
    config: str
    gleamdata: str
    eb_ms: str


def init_data(temp_dir: Path):
    with tarfile.open(MS_TAR, "r:*") as tar:
        tar.extractall(path=temp_dir)
        ms_name = tar.getnames()[0].split("/")[0]
        ms_path = (temp_dir / Path(ms_name)).as_posix()

    with tarfile.open(OSKAR_MOCK_TAR, "r:*") as tar:
        tar.extractall(path=temp_dir)
        eb_ms_name = tar.getnames()[0].split("/")[0]
        eb_ms = (temp_dir / Path(eb_ms_name)).as_posix()

    return TestResource(ms_path, CONFIG_PATH, GLEAMDATA, eb_ms)
