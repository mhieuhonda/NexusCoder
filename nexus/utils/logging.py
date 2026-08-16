"""Logging utilities cho Nexus Coder."""
import logging
import sys
from typing import Optional


def get_logger(name: str = "nexus", level: int = logging.INFO) -> logging.Logger:
    """Tạo logger chuẩn cho Nexus."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
