from contextlib import contextmanager
from fastapi import HTTPException


@contextmanager
def HttpResponse(success_callback=None, fail_callback=None):
    try:
        yield
        if success_callback:
            success_callback()
    except Exception as e:
        if fail_callback:
            fail_callback(e)
        else:
            raise HTTPException(status_code=400, detail=str(e))
