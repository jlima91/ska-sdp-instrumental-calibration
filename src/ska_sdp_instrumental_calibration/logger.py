"""Module for logger."""

import logging

from ska_ser_logging import configure_logging


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Setup a new logger with given name.

    If there are no handlers set, this will call
    ``configure_logging`` function from ``ska_ser_logging``
    to set the SKA standardized logging format at the root level,
    and also the handlers.

    Parameters
    ----------
    name: str
        Namespace for the logger.
    level: int
        Integer representing standard logging levels

    Returns
    -------
    Logger
        Logger object.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        configure_logging(level=level)
        logger = logging.getLogger(name)

    return logger
