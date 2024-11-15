"""Module for logger."""

# copied from ska-sdp-distributed-self-cal-prototype for consistent logging

import logging


def setup_logger(name: str) -> logging.Logger:
    """Set up logger.

    Args:
        name: Namespace for the logger.

    Returns:
        logger: Logger object.
    """
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
