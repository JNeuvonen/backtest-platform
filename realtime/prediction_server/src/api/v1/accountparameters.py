from fastapi import APIRouter, Depends, Response, status
from context import HttpResponseContext
from middleware import api_key_auth
from schema.account import AccountQuery
from src.api.v1.request_types import BodyCreateAccount


router = APIRouter()


class RoutePaths:
    ACCOUNT = "/"


@router.get(RoutePaths.ACCOUNT, dependencies=[Depends(api_key_auth)])
async def route_get_root():
    with HttpResponseContext():
        accounts = AccountQuery.get_accounts()
        return {"data": accounts}


@router.post(RoutePaths.ACCOUNT, dependencies=[Depends(api_key_auth)])
async def route_post_root(body: BodyCreateAccount):
    with HttpResponseContext():
        id = AccountQuery.create_account(body.model_dump())
        return Response(
            content=f"{str(id)}",
            status_code=status.HTTP_200_OK,
            media_type="text/plain",
        )
