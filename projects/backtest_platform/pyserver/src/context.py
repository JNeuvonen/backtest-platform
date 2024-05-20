from contextlib import contextmanager
import logging
from fastapi import HTTPException
from log import get_logger


@contextmanager
def HttpResponseContext(endpoint_call_success_msg=""):
    try:
        yield
        logger = get_logger()
        if endpoint_call_success_msg != "":
            logger.log(
                endpoint_call_success_msg,
                logging.INFO,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
