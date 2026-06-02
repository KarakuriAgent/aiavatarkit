import logging


def setup_logging():
    logger = logging.getLogger("aiavatar")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s"))
        logger.addHandler(handler)
    return logger
