import logging
import os
import time


def setup_logger(name: str, level = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    stream_handler = logging.StreamHandler()

    # Create formatter and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(stream_handler)

    return logger