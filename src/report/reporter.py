# ** coding: utf-8 **
"""
@Author: Zhihong Li, Rongrong Lin
@date: 2025/10/10
@description: Reporter with structlog integration.
@version: 4.0
"""

import os
from pathlib import Path

import structlog


def mkdir(path: str) -> None:
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    path = path.strip()
    path = path.rstrip("\\")
    if not os.path.exists(path):
        os.makedirs(path)


is_debug = True


class Logger:

    def __init__(self, file_dir: str | None, file_name: str | None, is_debug: bool = True, mode: str = 'wt+'):
        self.logger = self._init_log(file_dir, file_name, is_debug)
        self._mode = mode
        if is_debug == True and file_name is not None:
            self.logger.warning("Debug mode is enabled, the output file will not be generated.")
        else:
            self.logger.info("Debug mode is disabled, the output file will be generated.")

    def _init_log(self, file_dir: str, file_name: str, is_debug: bool = True):
        structlog.configure(
            processors=[

                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(utc=False, fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(),
                # 或者使用KeyValueRenderer
            ],
            logger_factory=
            structlog.PrintLoggerFactory() if is_debug else
            structlog.WriteLoggerFactory(
                file=Path(file_dir + "/" + file_name).open(self._mode, encoding="utf-8")
            ),
            cache_logger_on_first_use=True

        )
        return structlog.get_logger()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__new__(cls)
        return cls._instance
