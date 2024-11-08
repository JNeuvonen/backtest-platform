from fastapi import APIRouter, Depends, Response, status

from prediction_server.middleware import api_key_auth
from prediction_server.context import HttpResponseContext
from common_python.pred_serv_models.cloudlog import CloudLogQuery, create_log
from datetime import timedelta

from prediction_server.api.v1.request_types import BodyCreateCloudLog

router = APIRouter()


class RoutePaths:
    LOG = "/"


@router.get(RoutePaths.LOG, dependencies=[Depends(api_key_auth)])
async def route_get_root():
    with HttpResponseContext():
        seven_days_ago = timedelta(days=7)
        logs = CloudLogQuery.get_recent_logs(seven_days_ago)
        return {"data": logs}


@router.post(RoutePaths.LOG, dependencies=[Depends(api_key_auth)])
async def route_post_root(body: BodyCreateCloudLog):
    with HttpResponseContext():
        id = create_log(body.message, body.level, body.source_program)
        return Response(
            content=f"{str(id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )
