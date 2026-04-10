import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LEVEL_MAP = {
    "debug":    logging.DEBUG,
    "info":     logging.INFO,
    "warning":  logging.WARNING,
    "error":    logging.ERROR,
    "critical": logging.CRITICAL,
}

def setup_logger(name: str, level: str = "info") -> logging.Logger:
    """
    Returns a logger that writes to both rotating main log and a dedicated error log

    Args:
        name: logger name -- pass __name__ from the calling module
        level: Minimum log level for the main log. Error log is always ERROR+. Default to INFO.

    Returns:
        logging.Logger
    """

    logger = logging.getLogger(name)

    if logger.handlers:  # avoid dubplicate handlers on re-import
        return logger
    
    logger.setLevel(LEVEL_MAP.get(level, logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # main log: rotate at 2MB, keeps 5 backups
    main_handler = RotatingFileHandler(
        filename=LOGS_DIR / "pipeline.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    main_handler.setLevel(logging.DEBUG)
    main_handler.setFormatter(fmt)

    # error log: ERROR and above only
    error_handler = RotatingFileHandler(
        filename=LOGS_DIR / "errors.log",
        maxBytes=1 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(fmt)

    # console logs for developement purposes
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(fmt)

    logger.addHandler(main_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger