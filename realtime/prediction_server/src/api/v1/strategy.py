from fastapi import APIRouter, Response, status
from context import HttpResponseContext
from request_types import BodyCreateStrategy


router = APIRouter()


class RoutePaths:
    STRATEGY = "/"


@router.get(RoutePaths.STRATEGY)
async def route_get_root():
    with HttpResponseContext():
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )


@router.post(RoutePaths.STRATEGY)
async def route_create_strategy(body: BodyCreateStrategy):
    with HttpResponseContext():
        return Response(
            content="OK", status_code=status.HTTP_200_OK, media_type="text/plain"
        )
