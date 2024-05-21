from typing import Dict
from fastapi import APIRouter, Depends, status, Response
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.user import UserQuery
from common_python.http_utils import verify_ip_whitelisted
from common_python.auth.oauth import oauth2_scheme, get_user_info
from fastapi.security import HTTPBearer
from analytics_server.api.v1.request_types import BodyCreateuser


router = APIRouter()
token_auth_scheme = HTTPBearer()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def route_get_root(token: str = Depends(oauth2_scheme)):
    user_info = get_user_info(token)
    with HttpResponse():
        return Response(
            content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
        )


@router.post(RoutePaths.ROOT, dependencies=[Depends(verify_ip_whitelisted)])
async def route_create_entry(body: BodyCreateuser):
    with HttpResponse():
        body_dict = body.model_dump()
        id = UserQuery.create_entry(body_dict)
        return Response(
            content=str(id),
            media_type="text/plain",
            status_code=status.HTTP_200_OK,
        )
