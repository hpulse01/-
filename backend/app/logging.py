from __future__ import annotations

import logging
from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    logger = logging.getLogger()
    logger.setLevel(level.upper())

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    logger.handlers = [handler]