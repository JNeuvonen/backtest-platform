from fastapi import APIRouter, Depends, status, Response
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.user import UserQuery
from common_python.http_utils import verify_ip_whitelisted
from analytics_server.api.v1.request_types import BodyCreateuser


router = APIRouter()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def route_get_root():
    with HttpResponse():
        return Response(
            content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
        )


@router.post(RoutePaths.ROOT, dependencies=[Depends(verify_ip_whitelisted)])
async def route_create_entry(body: BodyCreateuser):
    with HttpResponse():
        id = UserQuery.create_entry(body.model_dump())
        return Response(
            content=str(id),
            media_type="text/plain",
            status_code=status.HTTP_200_OK,
        )
