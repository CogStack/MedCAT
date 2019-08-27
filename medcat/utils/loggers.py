"""Loggers
"""
import logging

FORMAT = logging.Formatter('%(asctime)s %(app_name)s: %(message)s')
logging.basicConfig(filename='tmp_cat.log', level=logging.DEBUG)

def basic_logger(name=''):
    """ The most basic logger type, log is just:
    logger_name: <basic_info>

    name:  logger name
    """
    if name:
        name = "." + name

    return logging.getLogger('cat' + name)
