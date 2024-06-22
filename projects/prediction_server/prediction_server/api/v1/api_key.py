from fastapi import APIRouter
from common_python.pred_serv_models.api_key import APIKeyQuery
from prediction_server.context import HttpResponseContext


router = APIRouter()


class RoutePaths:
    API_KEY = "/"


@router.post(RoutePaths.API_KEY)
async def route_get_root():
    with HttpResponseContext():
        api_key = APIKeyQuery.gen_api_key()
        return {"data": api_key}
