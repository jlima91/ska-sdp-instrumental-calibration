"""SDM class to manage science data model"""

import shutil
from enum import Enum
from pathlib import Path
from typing import Optional


def prepare_qa_path(output_dir, sdm_path, **kwargs):
    """
    Initialize SDM directory structure and prepare the QA path.

    Parameters
    ----------
    output_dir : str
        Base directory used to construct the SDM path if not provided.
    sdm_path : str or None
        Path to the SDM directory. If None, it defaults to a 'sdm'
        subdirectory within output_dir.
    **kwargs : dict
        Additional keyword arguments for path preparation.

    Returns
    -------
    str
        The path to the prepared log directory.
    """
    if sdm_path is None:
        sdm_path = f"{output_dir}/sdm/"

    SDM.init(sdm_path)
    return SDM.prepare_log_dir(sdm_path, "inst")


class SDM(Enum):
    """SDM management functions"""

    SKY = "sky"
    TELMODEL = "telmodel"
    INSTRUMENT = "telmodel/instrument"
    GAINS = "calibration/gains"
    POINTING = "calibration/pointing"
    BANDPASS = "calibration/bandpass"
    LOGS = "logs"

    @staticmethod
    def _get_current_model_count(
        sdm_path: Path, pattern: Optional[str] = None
    ) -> int:
        """
        Calculate the highest numeric prefix among existing model folders.

        Parameters
        ----------
        sdm_path : Path
            Path to the directory containing model folders.
        pattern : str, optional
            Glob pattern to filter candidate directories.

        Returns
        -------
        int
            The maximum index found, or 0 if no indexed folders exist.
        """
        candidates = sdm_path.glob(pattern) if pattern else sdm_path.iterdir()
        folder_counters = [
            int(idx)
            for _sdm in candidates
            if (idx := _sdm.name.split("-")[0]).isdigit()
        ]
        return 0 if not folder_counters else max(folder_counters)

    def find_model(
        self,
        sdm_root: str,
        field_id: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Retrieve model file based on type, field, and pattern.

        Parameters
        ----------
        sdm_root : str
            The root directory of the SDM.
        field_id : str, optional
            Subdirectory for a specific field ID.
        pattern : str, optional
            Glob pattern to filter files.

        Returns
        -------
        Path
            Return the first matching path
        """
        models = self.find_models(sdm_root, field_id, pattern)
        return models[0] if len(models) > 0 else None

    def find_models(
        self,
        sdm_root: str,
        field_id: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[Path]:
        """
        Retrieve model files based on type, field, and pattern.

        Parameters
        ----------
        sdm_root : str
            The root directory of the SDM.
        field_id : str, optional
            Subdirectory for a specific field ID.
        pattern : str, optional
            Glob pattern to filter files.

        Returns
        -------
        list of Path
            A list of paths matching the search criteria.
        """
        root = Path(sdm_root)
        sdm_path = root / self.value

        if field_id is None:
            pattern = pattern or "*"
            return list(sdm_path.rglob(pattern))

        sdm_path = sdm_path / field_id

        if pattern is not None:
            return list(sdm_path.rglob(pattern))

        return list(sdm_path.iterdir())

    def prepare_model(
        self, sdm_root: str, field_id: str, model_name: str
    ) -> Path:
        """
        Prepare the directory for a new model, versioning old ones if present.

        Parameters
        ----------
        sdm_root : str
            The root directory of the SDM.
        field_id : str
            The specific field ID directory.
        model_name : str
            The name of the model to prepare.

        Returns
        -------
        Path
            The path to the prepared model directory.

        Raises
        ------
        RuntimeError
            If called on SDM.LOGS.
        TypeError
            If model_name is None.
        """
        if self == SDM.LOGS:
            raise RuntimeError(
                "Use SDM.prepare_log_dir to prepare log directory"
            )

        root = Path(sdm_root) / self.value
        if model_name is None:
            raise TypeError("Model name not provided")

        sdm_path = root / field_id
        if not sdm_path.exists():
            sdm_path.mkdir(parents=True, exist_ok=True)

        model_path = sdm_path / model_name

        if model_path.exists():
            next_count = (
                self._get_current_model_count(sdm_path, f"*{model_name}") + 1
            )
            new_path = sdm_path / f"{next_count:02}-{model_name}"
            shutil.move(model_path, new_path)

        return model_path

    @classmethod
    def prepare_log_dir(cls, sdm_root, pipeline) -> Path:
        """
        Create a versioned log directory for a specific pipeline.

        Parameters
        ----------
        sdm_root : str
            The root directory of the SDM.
        pipeline : str
            The name of the pipeline requesting the log directory.

        Returns
        -------
        Path
            The path to the newly created versioned log directory.
        """
        root = Path(sdm_root) / cls.LOGS.value
        next_count = cls._get_current_model_count(root) + 1
        idx_pipeline = f"{next_count:02}-{pipeline}"

        sdm_path = root / idx_pipeline
        sdm_path.mkdir(parents=True, exist_ok=True)
        return sdm_path

    @classmethod
    def get_log_dir(cls, sdm_root, pipeline) -> Path:
        """
        Gets the most recent log folder for the pipeline.
        If a log folder does not exist, it creates a new log folder
        and return that.

        Parameters
        ----------
        sdm_root : str
            The root directory of the SDM.
        pipeline : str
            The name of the pipeline requesting the log directory.

        Returns
        -------
        Path
            The path to the most recent log directory for the pipeline.
        """
        log_path = Path(sdm_root) / cls.LOGS.value
        folder_counters = {
            int(idx): _sdm.name
            for _sdm in log_path.iterdir()
            if (idx := _sdm.name.split("-")[0]).isdigit()
            and pipeline == _sdm.name.split("-")[1]
        }

        if not folder_counters:
            return cls.prepare_log_dir(sdm_root, pipeline)

        return log_path / folder_counters[max(folder_counters)]

    @classmethod
    def init(cls, sdm_root: str):
        """
        Initialize the SDM directory structure.

        Parameters
        ----------
        sdm_root : str
            The root directory path where SDM folders will be created.
        """
        root = Path(sdm_root)

        for sdm in cls:
            sdm_path = root / sdm.value
            sdm_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clone(exiting_sdm_root: str | Path, new_sdm_root: str | Path):
        """
        Clone an existing SDM root directory to a new location.

        Parameters
        ----------
        exiting_sdm_root : str or Path
            Path to the source SDM root directory to be copied.
        new_sdm_root : str or Path
            Path where the SDM root directory will be created.
        """
        shutil.copytree(exiting_sdm_root, new_sdm_root)
