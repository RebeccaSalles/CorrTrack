"""Logging: pipeline steps and checkpoints, to the console and/or a file.

A single logger named "corrtrack", configured once per process (the first call
fixes the level and the file). The console writes to stderr (the result stays
on stdout); `--log-file` additionally writes a `.log`.

    log = get_logger("info", "run.log")
    log.info("…")          # step
    log.debug("…")         # detail (fine-grained checkpoints)
"""

import logging
import os
import sys

_LOGGER = None
_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO,
           "warning": logging.WARNING, "error": logging.ERROR}


def get_logger(level="info", log_file=""):
    """The "corrtrack" logger (configured on the 1st call; reused afterwards)."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("corrtrack")
    logger.setLevel(_LEVELS.get(str(level).lower(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        logger.info("logging to file %s", log_file)

    _LOGGER = logger
    return logger


def reset():
    """Reset (useful in tests / multiple calls within the same process)."""
    global _LOGGER
    _LOGGER = None
