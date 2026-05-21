"""
logger.py
─────────────────────────────────────────────────────────────────────
logging configuration for Portfolio Analytics Dashboard.
 
Streamlit Cloud edition — stdout only (no file handler).
Logs are visible in the Streamlit Cloud dashboard under:
  Manage App → Logs
 
Usage (in any module):
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Data loaded successfully")
─────────────────────────────────────────────────────────────────────
"""
 
import logging
import sys
 
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
 
 
def _build_formatter() -> logging.Formatter:
    """Return a shared formatter used by all handlers."""
    return logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
 
 
def _console_handler() -> logging.StreamHandler:
    """
    Stream handler that writes to stdout.
    On Streamlit Cloud, stdout is captured and shown in the
    platform's Logs panel — this is the only handler needed.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_build_formatter())
    return handler
 
 
def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger with a stdout handler.
 
    Parameters
    ----------
    name  : Logger name — pass __name__ from the calling module so log
            records show exactly which file emitted them.
    level : Root level for this logger (default DEBUG captures everything).
 
    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
 
    # Avoid adding duplicate handlers when Streamlit hot-reloads the module
    if not logger.handlers:
        logger.setLevel(level)
        logger.addHandler(_console_handler())
        # Prevent log records from bubbling up to the root logger
        logger.propagate = False
 
    return logger
 
 
# ── Module-level logger ───────────────────────────────────────────────
_log = get_logger(__name__)
_log.debug("logger.py initialised — running on Streamlit Cloud (stdout only)")