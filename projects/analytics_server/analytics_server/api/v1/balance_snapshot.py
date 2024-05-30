from common_python.auth.oauth2 import verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.balance_snapshot import BalanceSnapshotQuery
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    ROOT = "/"
    LATEST = "/latest"
    ONE_DAY_INTERVAL = "/1d-inteval"


@router.get(RoutePaths.ROOT)
async def get_balance_snapshots(token: str = Depends(verify_access_token)):
    with HttpResponse():
        data = BalanceSnapshotQuery.read_all_entries()
        return {"data": data}


@router.get(RoutePaths.LATEST)
async def get_latest_balance_snapshot(token: str = Depends(verify_access_token)):
    with HttpResponse():
        data = BalanceSnapshotQuery.get_last_snapshot()
        return {"data": data}


@router.get(RoutePaths.ONE_DAY_INTERVAL)
async def get_balance_snapshots_one_day_interval(
    token: str = Depends(verify_access_token),
):
    with HttpResponse():
        data = BalanceSnapshotQuery.get_snapshots_with_one_day_interval()
        return {"data": data}
