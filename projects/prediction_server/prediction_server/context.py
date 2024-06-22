from contextlib import contextmanager
from fastapi import HTTPException
from prediction_server.log import get_logger


@contextmanager
def HttpResponseContext(endpoint_call_success_msg=""):
    try:
        yield
        logger = get_logger()
        if endpoint_call_success_msg != "":
            logger.info(
                endpoint_call_success_msg,
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
