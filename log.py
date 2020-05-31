import logging
from time import gmtime, strftime
import os

__all__ = ['debug', 'info', 'warning', 'error', 'critical', 'is_debug', 'set_debug']

REMOVE_LOGFILE_IF_EXIST = True
LOG_LEVEL = logging.INFO

logger = logging.getLogger("HSS")


def init(save_file_log, logs_dir, log_name):
    global logger

    time_postfix = log_name  ##strftime("%d_%m_%Y,%H:%M:%S", gmtime())
    logs_name = logs_dir + '/' + time_postfix + '.log'

    if REMOVE_LOGFILE_IF_EXIST and os.path.exists(logs_name):
        os.remove(logs_name)

    fh = logging.FileHandler(logs_name) if save_file_log else logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(LOG_LEVEL)


def set_debug():
    global LOG_LEVEL, logger
    LOG_LEVEL = logging.DEBUG
    if logger:
        logger.setLevel(LOG_LEVEL)


def is_debug():
    return LOG_LEVEL == logging.DEBUG


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