from fastapi.security import APIKeyHeader
from config import is_dev
from schema.api_key import APIKeyQuery
from schema.whitelisted_ip import WhiteListedIPQuery
from fastapi import Request, Security, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

api_key_header = APIKeyHeader(name="X-API-KEY")


async def api_key_auth(api_key: str = Security(api_key_header)):
    if not is_dev() and not APIKeyQuery.is_valid_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ValidateIPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.client is None:
            return JSONResponse(status_code=403, content={"detail": "IP not allowed"})

        ip = request.client.host
        if not is_dev() and not WhiteListedIPQuery.is_allowed_ip(ip):
            return JSONResponse(status_code=403, content={"detail": "IP not allowed"})
        response = await call_next(request)
        return response
