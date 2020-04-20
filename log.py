import logging

__all__ = ['debug', 'info', 'warning', 'error', 'critical']

LOG_LEVEL = logging.DEBUG
SAVE_FILE_LOG = False

logger = logging.getLogger("HSS")
fh = logging.FileHandler("hss.log") if SAVE_FILE_LOG else logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)


def critical(msg):
    logger.critical(msg)


def error(msg):
    logger.error(msg)


def warning(msg):
    logger.warning(msg)


def info(msg):
    logger.info(msg)


def debug(msg):
    logger.debug(msg)