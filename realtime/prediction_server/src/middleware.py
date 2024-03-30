from config import is_dev
from schema.whitelisted_ip import WhiteListedIPQuery
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class ValidateIPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.client is None:
            return JSONResponse(status_code=403, content={"detail": "IP not allowed"})

        ip = request.client.host
        if is_dev() is False and not WhiteListedIPQuery.is_allowed_ip(ip):
            return JSONResponse(status_code=403, content={"detail": "IP not allowed"})
        response = await call_next(request)
        return response
