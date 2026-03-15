# -*- coding: utf-8 -*-
"""统一日志，替代 print，便于记录特征工程、训练、调参关键步骤。"""
import logging
import sys
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(
    name: str = "fracture_ml",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    logger.addHandler(h)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    _LOGGER = logger
    return logger
