"""
Dataclasses for the sim.yaml configuration file, with validation.

The file has a common section (frequency and time axes, fields) and one optional section per script ("get_data",
"generate_gaintable", "generate_tec" and "simulate_visibility"). Each section
is loaded into a dataclass:

- Unknown keys are rejected, so that a typo does not silently fall back to
  a default value.
- Values are checked against the type annotation of the field. Floats also
  accept strings, because PyYAML reads numbers like "50.0e6" (exponent
  without a sign) as strings.
- Values that depend on each other are checked in __post_init__.

This module only depends on the standard library and PyYAML, because it is
also imported by lib/run_oskar.py, possibly inside the OSKAR Singularity
image.
"""

from __future__ import annotations

import dataclasses
import os
import re
import typing
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Options of "telescope/station_type" in OSKAR
STATION_TYPES = ("Aperture array", "Isotropic beam", "Gaussian beam", "VLA (PBCOR)")

# Options of "telescope/aperture_array/element_pattern/taper/type" in OSKAR
ELEMENT_TAPERS = ("None", "Cosine", "Gaussian")

# Bright sources which lib/run_oskar.py can add to the sky model
BRIGHT_SOURCES = ("centaurus_a", "fornax_a", "taurus_a")

# Corruption files which OSKAR reads from the telescope model, by the key of
# the "simulate_visibility" section which sets them. The telescope model must
# not contain them: they are linked into the copy of the model only if set.
TEL_MODEL_CORRUPTION_FILES = {
    "gaintable": "gain_model.h5",
    "cable_delay": "cable_length_error.txt",
}


# Top-level layout files of a telescope model, in the order OSKAR looks for
# them. OSKAR has one station per row of the first one found.
TEL_MODEL_LAYOUT_FILES = (
    "layout.txt",
    "layout_enu.txt",
    "layout_ecef.txt",
    "layout_wgs84.txt",
)


# Name of the MS of a scan written by lib/run_oskar.py, and the pattern which
# postprocess_visibility.py uses to read the scan number back from it
MS_NAME_FORMAT = "visibility.scan-{scan_number}.ms"
MS_NAME_PATTERN = re.compile(r"visibility\.scan-(?P<scan_number>\d+)\.ms")


class ConfigError(ValueError):
    """Raised when sim.yaml is not valid."""


def count_stations(tel_model: str) -> int:
    """
    Number of stations of an OSKAR telescope model, as counted by OSKAR: the
    rows of the top-level layout file, without blank and comment ("#")
    lines. This is not the number of station directories, which can be
    shared by several stations (station_type_map.txt).
    """
    for name in TEL_MODEL_LAYOUT_FILES:
        path = os.path.join(tel_model, name)
        if os.path.isfile(path):
            with open(path, "r") as f:
                rows = [line for line in f if line.strip()[:1] not in ("", "#")]
            if not rows:
                raise ConfigError(f"No stations in {path}")
            return len(rows)
    raise ConfigError(
        f"No layout file in the telescope model {tel_model} (looked for "
        f"{', '.join(TEL_MODEL_LAYOUT_FILES)})"
    )


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Check (and convert where it is safe) a value against a type hint."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        # Only Optional[X] is used
        if value is None:
            return None
        (inner,) = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        return _coerce(value, inner, where)
    if origin is dict:
        key_type, value_type = typing.get_args(hint)
        if not isinstance(value, dict) or not value:
            raise ConfigError(f"{where}: expected a non-empty mapping")
        return {
            _coerce(k, key_type, where): _coerce(v, value_type, f"{where}.{k}")
            for k, v in value.items()
        }
    if origin is list:
        (item_type,) = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list, got {value!r}")
        return [_coerce(v, item_type, f"{where}[{i}]") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(hint):
        return _from_dict(hint, value, where)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected a number, got {value!r}") from None
    if hint is str:
        # PyYAML reads an unquoted "2000-01-03 22:33:30" as a datetime
        if isinstance(value, (date, datetime)):
            return str(value)
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise TypeError(f"Unsupported type hint {hint!r} for {where}")


def _from_dict(cls, data: Any, where: str):
    """Create the dataclass `cls` from a mapping, with validation."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {data!r}")

    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{where}: unknown keys {unknown}. Allowed keys: {sorted(known)}"
        )
    missing = sorted(
        name
        for name, f in known.items()
        if name not in data
        and f.default is dataclasses.MISSING
        and f.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ConfigError(f"{where}: missing required keys {missing}")

    hints = typing.get_type_hints(cls)
    kwargs = {
        name: _coerce(value, hints[name], f"{where}.{name}")
        for name, value in data.items()
    }
    try:
        return cls(**kwargs)
    except ConfigError as err:
        raise ConfigError(f"{where}: {err}") from None


def _check(condition: bool, message: str):
    if not condition:
        raise ConfigError(message)


@dataclass
class Target:
    """A target in "fields", observed centred on its transit time."""

    ra_deg: float
    dec_deg: float
    scan_id_start: int
    transit_time: str
    # Written into the MS by postprocess_visibility.py: FIELD NAME (default:
    # the target name) and STATE OBS_MODE (default: kept as written by OSKAR,
    # "OBSERVE_TARGET#ON_SOURCE")
    field_id: Optional[str] = None
    scan_intent: Optional[str] = None

    def __post_init__(self):
        _check(-90.0 <= self.dec_deg <= 90.0, "dec_deg must be in [-90, 90]")
        try:
            datetime.fromisoformat(self.transit_time)
        except ValueError:
            raise ConfigError(
                f"transit_time {self.transit_time!r} is not an ISO format date"
            ) from None

    @property
    def transit_datetime(self) -> datetime:
        return datetime.fromisoformat(self.transit_time)

    def scan_start_time(
        self, obs_length_mins: float, scan_index: int = 0, num_scans: int = 1
    ) -> datetime:
        """
        Start time of a scan. The whole observation (all scans) is centred
        on the transit time.
        """
        scan_length_mins = obs_length_mins / num_scans
        return self.transit_datetime + (
            timedelta(minutes=scan_index * scan_length_mins)
            - timedelta(minutes=0.5 * obs_length_mins)
        )

    def scan_number(self, scan_index: int = 0) -> int:
        """Scan number of a scan, used in the name of its MS."""
        return self.scan_id_start + scan_index


@dataclass
class OutlierConfig:
    """Gain outliers injected by generate_gaintable.py."""

    enable: bool = False
    amp_min: float = 2.0
    amp_max: float = 5.0
    n_stations_to_corrupt: int = 20
    n_channels_to_corrupt: int = 100

    def __post_init__(self):
        _check(self.amp_min <= self.amp_max, "amp_min must be <= amp_max")
        _check(self.n_stations_to_corrupt >= 0, "n_stations_to_corrupt must be >= 0")
        _check(self.n_channels_to_corrupt >= 0, "n_channels_to_corrupt must be >= 0")


@dataclass
class GenerateGaintableConfig:
    """The "generate_gaintable" section."""

    output_gaintable: str
    bandpass_spline_file: str
    outlier_config: OutlierConfig = field(default_factory=OutlierConfig)
    station_offset: bool = True
    time_variant: bool = True
    rfi: bool = False
    rfi_start_freq_hz: Optional[float] = None
    rfi_end_freq_hz: Optional[float] = None
    plot: bool = False
    plot_output_dir: str = "./gaintable_generation_plots"

    def __post_init__(self):
        if self.rfi:
            _check(
                self.rfi_start_freq_hz is not None and self.rfi_end_freq_hz is not None,
                "rfi_start_freq_hz and rfi_end_freq_hz are required if rfi is true",
            )
            _check(
                self.rfi_start_freq_hz < self.rfi_end_freq_hz,
                "rfi_start_freq_hz must be < rfi_end_freq_hz",
            )


@dataclass
class TecLayer:
    """A layer of the ionosphere simulated by generate_tec.py."""

    r0_m: float
    speed_m_per_s: float
    direction_deg: float
    height_m: float

    def __post_init__(self):
        _check(self.r0_m > 0, "r0_m must be > 0")
        _check(self.speed_m_per_s >= 0, "speed_m_per_s must be >= 0")
        _check(self.height_m > 0, "height_m must be > 0")


def _default_tec_layers() -> List[TecLayer]:
    speed = 150e3 / 3600.0  # 150 km/h
    return [
        TecLayer(r0_m=20e3, speed_m_per_s=speed, direction_deg=60.0, height_m=300e3),
        TecLayer(
            r0_m=20e3, speed_m_per_s=speed / 2, direction_deg=-30.0, height_m=310e3
        ),
    ]


@dataclass
class GenerateTecConfig:
    """
    The "generate_tec" section. The frequency, duration and target of the
    screen come from the common section.
    """

    output_tec_screen: str
    # ARatmospy folder, added to sys.path. If not set, ArScreens must be
    # importable (e.g. with PYTHONPATH).
    aratmospy_dir: Optional[str] = None
    screen_width_m: float = 200e3
    subaperture_size_m: float = 20e3
    pixel_size_m: float = 100.0
    # Time between screens. Default: sampling_time_sec
    time_interval_sec: Optional[float] = None
    alpha_mag: float = 0.999
    target_rms_tecu: float = 0.01
    random_seed: int = 42
    layers: List[TecLayer] = field(default_factory=_default_tec_layers)

    def __post_init__(self):
        _check(self.pixel_size_m > 0, "pixel_size_m must be > 0")
        _check(
            self.pixel_size_m <= self.subaperture_size_m <= self.screen_width_m,
            "must have pixel_size_m <= subaperture_size_m <= screen_width_m",
        )
        _check(
            self.time_interval_sec is None or self.time_interval_sec > 0,
            "time_interval_sec must be > 0",
        )
        _check(0 < self.alpha_mag < 1, "alpha_mag must be in (0, 1)")
        _check(self.target_rms_tecu > 0, "target_rms_tecu must be > 0")
        _check(len(self.layers) > 0, "layers must not be empty")


@dataclass
class OskarImagerConfig:
    """Images made with the OSKAR imager by lib/run_oskar.py."""

    make_image: bool = False
    make_psf: bool = False
    image_size: int = 2980
    cellsize_arcsec: float = 21.66

    def __post_init__(self):
        _check(self.image_size > 0, "image_size must be > 0")
        _check(self.cellsize_arcsec > 0, "cellsize_arcsec must be > 0")


@dataclass
class SimulateVisibilityConfig:
    """
    The "simulate_visibility" section, also read by lib/run_oskar.py. Paths are
    relative to the current directory.
    """

    # Telescope model directory (required). Copied into the output directory
    # for each run, and the copy is given to OSKAR.
    tel_model: str

    # OSKAR Singularity image. If not set, lib/run_oskar.py is run with the
    # current Python, which must be able to import oskar.
    oskar_sif: Optional[str] = None

    # Sky models
    gleam_file: Optional[str] = None
    lotss_file: Optional[str] = None
    field_radius_deg: float = 10.0
    sky_model_file: Optional[str] = None
    bright_sources: List[str] = field(default_factory=list)

    # Station beam
    station_type: str = "Aperture array"
    element_taper: str = "None"
    element_cosine_power: float = 1.0
    element_gaussian_fwhm_deg: float = 45.0
    no_smearing: bool = False

    # OSKAR simulator
    double_precision: bool = False
    use_gpus: bool = False
    add_noise: bool = False

    # Corruptions
    gaintable: Optional[str] = None
    cable_delay: Optional[str] = None
    tec_screen: Optional[str] = None

    # Dirty image with DP3 and wsclean
    create_dirty_image: bool = False
    image_size: int = 1024
    pixel_size: str = "2arcsec"

    oskar_imager: OskarImagerConfig = field(default_factory=OskarImagerConfig)

    run_oskar_extra_params: Optional[str] = None

    def __post_init__(self):
        _check(self.field_radius_deg > 0, "field_radius_deg must be > 0")
        _check(
            self.station_type in STATION_TYPES,
            f"station_type must be one of {STATION_TYPES}",
        )
        _check(
            self.element_taper in ELEMENT_TAPERS,
            f"element_taper must be one of {ELEMENT_TAPERS}",
        )
        _check(
            self.element_gaussian_fwhm_deg > 0,
            "element_gaussian_fwhm_deg must be > 0",
        )
        _check(self.image_size > 0, "image_size must be > 0")
        unknown = sorted(set(self.bright_sources) - set(BRIGHT_SOURCES))
        _check(not unknown, f"bright_sources {unknown} not in {BRIGHT_SOURCES}")

    def input_dirs(self) -> Dict[str, str]:
        """Input directories which must exist, by key name."""
        return {"tel_model": self.tel_model}

    def input_files(self) -> Dict[str, Optional[str]]:
        """Input files which must exist, by key name (None if not used)."""
        return {
            "oskar_sif": self.oskar_sif,
            "gleam_file": self.gleam_file,
            "lotss_file": self.lotss_file,
            "sky_model_file": self.sky_model_file,
            "gaintable": self.gaintable,
            "cable_delay": self.cable_delay,
            "tec_screen": self.tec_screen,
        }

    def with_resolved_paths(self) -> "SimulateVisibilityConfig":
        """
        Copy with the input paths (files and directories) made absolute, from
        the current directory, and symlinks resolved.
        """
        paths = {**self.input_dirs(), **self.input_files()}
        return dataclasses.replace(
            self,
            **{key: os.path.realpath(path) for key, path in paths.items() if path},
        )


@dataclass
class GetDataConfig:
    """The "get_data" section."""

    s3_prefix: str = "s3://skao-sdp-testdata/PI27-Low-G4"

    def __post_init__(self):
        _check(self.s3_prefix.startswith("s3://"), "s3_prefix must start with s3://")


@dataclass
class SimConfig:
    """The whole sim.yaml file."""

    scenario: str
    simulation_start_frequency_hz: float
    simulation_end_frequency_hz: float
    correlated_channel_bandwidth_hz: float
    observing_time_mins: float
    sampling_time_sec: float
    fields: Dict[str, Dict[str, Target]]
    get_data: GetDataConfig = field(default_factory=GetDataConfig)
    generate_gaintable: Optional[GenerateGaintableConfig] = None
    generate_tec: Optional[GenerateTecConfig] = None
    simulate_visibility: Optional[SimulateVisibilityConfig] = None

    def __post_init__(self):
        _check(
            self.simulation_start_frequency_hz < self.simulation_end_frequency_hz,
            "simulation_start_frequency_hz must be < simulation_end_frequency_hz",
        )
        _check(
            self.correlated_channel_bandwidth_hz > 0,
            "correlated_channel_bandwidth_hz must be > 0",
        )
        _check(self.observing_time_mins > 0, "observing_time_mins must be > 0")
        _check(self.sampling_time_sec > 0, "sampling_time_sec must be > 0")

    @classmethod
    def from_dict(cls, data: Any, where: str = "sim.yaml") -> "SimConfig":
        return _from_dict(cls, data, where)

    @classmethod
    def load(cls, path) -> "SimConfig":
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f), str(path))

    def first_target(self) -> Tuple[str, str, Target]:
        """
        The first target of the first field, as (field name, target name,
        target). Only this one is simulated by simulate_visibility.py.
        """
        field_name, targets = next(iter(self.fields.items()))
        target_name, target = next(iter(targets.items()))
        return field_name, target_name, target

    def get_target(self, field_name: str, target_name: str) -> Target:
        if field_name not in self.fields:
            raise ConfigError(f"Field name '{field_name}' not known")
        if target_name not in self.fields[field_name]:
            raise ConfigError(f"Target name '{target_name}' not known")
        return self.fields[field_name][target_name]

    def require_simulate_visibility(self) -> SimulateVisibilityConfig:
        if self.simulate_visibility is None:
            raise ConfigError("sim.yaml has no 'simulate_visibility' section")
        return self.simulate_visibility

    @property
    def n_stations(self) -> int:
        """Number of stations of the telescope model simulate_visibility.tel_model."""
        return count_stations(self.require_simulate_visibility().tel_model)

    def require_generate_gaintable(self) -> GenerateGaintableConfig:
        if self.generate_gaintable is None:
            raise ConfigError("sim.yaml has no 'generate_gaintable' section")
        gt_cfg = self.generate_gaintable
        if gt_cfg.outlier_config.enable:
            n_stations = self.n_stations
            _check(
                gt_cfg.outlier_config.n_stations_to_corrupt <= n_stations,
                "generate_gaintable.outlier_config.n_stations_to_corrupt must be "
                f"<= the number of stations of simulate_visibility.tel_model "
                f"({n_stations})",
            )
        if gt_cfg.rfi:
            _check(
                self.simulation_start_frequency_hz
                <= gt_cfg.rfi_start_freq_hz
                < gt_cfg.rfi_end_freq_hz
                <= self.simulation_end_frequency_hz,
                "generate_gaintable.rfi_start_freq_hz and rfi_end_freq_hz must "
                "be within simulation_start_frequency_hz and "
                "simulation_end_frequency_hz",
            )
        return gt_cfg

    def require_generate_tec(self) -> GenerateTecConfig:
        if self.generate_tec is None:
            raise ConfigError("sim.yaml has no 'generate_tec' section")
        return self.generate_tec
