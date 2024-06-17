import logging
import multiprocessing
from log import get_logger


log_event_queue: multiprocessing.Queue = multiprocessing.Queue()


def log_event_loop_queue(queue):
    while True:
        logger = get_logger()
        item = queue.get()
        if item is not None:
            logger.log(
                item,
                logging.INFO,
                True,
                True,
            )
