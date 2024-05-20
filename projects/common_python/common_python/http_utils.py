from contextlib import contextmanager
from fastapi import HTTPException, Request

from common_python.pred_serv_models.whitelisted_ip import WhiteListedIPQuery
from common_python.server_config import is_dev


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


async def verify_ip_whitelisted(request: Request):
    if request.client is None:
        raise HTTPException(status_code=403, detail="IP not allowed")

    ip = request.client.host
    if not is_dev() and not WhiteListedIPQuery.is_allowed_ip(ip):
        raise HTTPException(status_code=403, detail="IP not allowed")

    return request
