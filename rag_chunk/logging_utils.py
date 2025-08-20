# -*- coding: utf-8 -*-
"""Loguru-based logging utilities.

中文说明：
    - 统一初始化 loguru，输出到控制台与文件。
"""

from __future__ import annotations

import os
from loguru import logger


def init_logger(work_dir: str) -> None:
    """Initialize loguru handlers.

    Args:
        work_dir: 工作目录，用于存放日志文件。
    """
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True, level="INFO")
    logger.add(
        os.path.join(work_dir, "logs", "run.log"),
        rotation="5 MB",
        retention=5,
        level="DEBUG",
        encoding="utf-8",
    )
    logger.info("Logger initialized. Work dir: {}", work_dir)