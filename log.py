import logging
from time import gmtime, strftime
import os
__all__ = ['debug', 'info', 'warning', 'error', 'critical', 'is_debug']

LOG_LEVEL = logging.INFO
SAVE_FILE_LOG = True

logger = logging.getLogger("HSS")

time_postfix = 'log' ##strftime("%d_%m_%Y,%H:%M:%S", gmtime())
logs_name = 'logs/' + time_postfix + '.log'

if os.path.exists(logs_name):
    os.remove(logs_name)

fh = logging.FileHandler(logs_name) if SAVE_FILE_LOG else logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(LOG_LEVEL)

def is_debug():
    return LOG_LEVEL != logging.INFO


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