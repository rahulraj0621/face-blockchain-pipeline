"""
utils/logger.py – Coloured console logger used across all pipeline stages.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent coloured format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Simple format: [LEVEL] name – message
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  →  %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
