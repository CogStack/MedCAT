"""Loggers
"""
import sys
import logging

def basic_logger(name, config):
    """ The most basic logger type, log is just:
    logger_name: <basic_info>

    name:  logger name
    """
    name = "medcat." + name

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # Base logger and level

    if len(logger.handlers) == 0: # If we do not have any handlers add them
        # create a file handler
        fh = logging.FileHandler('medcat.log')
        fh.setLevel(config.general['log_level'])
        # create console handler 
        ch = logging.StreamHandler()
        ch.setLevel(config.general['log_level'])

        # create formatter and add it to the handlers
        formatter = logging.Formatter(config.general['log_format'])
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        # Change the existing loggers
        for handler in logger.handlers:
            handler.setLevel(config.general['log_level'])
            formatter = logging.Formatter(config.general['log_format'])
            handler.setFormatter(formatter)

    return logger
