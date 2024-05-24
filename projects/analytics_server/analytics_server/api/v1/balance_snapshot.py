from common_python.auth.oauth2 import verify_access_token
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def get_balance_snapshots(token: str = Depends(verify_access_token)):
    pass
