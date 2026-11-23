"""
Gaintables written by the run, read back for the QA report.
"""

from functools import lru_cache

from ...data_managers.h5parm import read_h5parm_gains
from ..exceptions import RunMetadataError


# maxsize=1: fields are reported one after another, and each of their
# queries reads the same final gaintable, so only the last read is reused.
# A bigger cache would keep the gains of every field in memory
@lru_cache(maxsize=1)
def read_gaintable(gaintable_path: str) -> tuple:
    """
    Read a gaintable, once for consecutive queries of the same field.

    The arrays returned are shared by every caller of the same path, so
    they must not be modified in place.

    Parameters
    ----------
    gaintable_path : str
        Path of the H5parm gaintable.

    Returns
    -------
    tuple
        ``(gain, weight, solved, soltab)`` as returned by
        :func:`read_h5parm_gains`, with an antenna axis on ``soltab``.

    Raises
    ------
    RunMetadataError
        If the gaintable was not produced, cannot be read or has no
        antenna axis. Errors are not cached.
    """
    try:
        gains = read_h5parm_gains(gaintable_path)
    except FileNotFoundError as error:
        # The run metadata lists the products a run should write, so a
        # listed gaintable can be missing if its export failed
        raise RunMetadataError(
            f"Final gaintable {gaintable_path} was not produced"
        ) from error
    except (OSError, KeyError, ValueError) as error:
        # A bad product, not a bug: a truncated or corrupt HDF5 file
        # (OSError), no solset (KeyError) or no amplitude or phase soltab
        # (ValueError)
        raise RunMetadataError(
            f"Final gaintable {gaintable_path} cannot be read: {error}"
        ) from error

    # Every query reads its stations off the antenna axis
    *_, soltab = gains
    if soltab.antenna is None:
        raise RunMetadataError(
            f"Gaintable {gaintable_path} has no antenna axis"
        )

    return gains
