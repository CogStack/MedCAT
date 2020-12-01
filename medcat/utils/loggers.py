"""Loggers
"""
import sys
import logging

def add_handlers(log):
    """ The most basic logger type, log is just:
    logger_name: <basic_info>

    name:  logger name
    """
    if len(log.handlers) == 0: # If we do not have any handlers add them
        # create a file handler
        fh = logging.FileHandler('medcat.log')
        # create console handler 
        ch = logging.StreamHandler()

        # add the handlers to the logger
        log.addHandler(fh)
        log.addHandler(ch)

    return log
