from fastapi import APIRouter, status, Response
from common_python.http_utils import HttpResponse


router = APIRouter()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def route_get_root():
    with HttpResponse():
        return Response(
            content="Curious?", media_type="text/plain", status_code=status.HTTP_200_OK
        )
