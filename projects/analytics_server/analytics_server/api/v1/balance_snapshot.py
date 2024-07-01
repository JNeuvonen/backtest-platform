from common_python.auth.oauth2 import verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.balance_snapshot import BalanceSnapshotQuery
from common_python.binance.funcs import (
    get_account_assets_state,
    repay_loan_with_available_balance,
)
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    ROOT = "/"
    LATEST = "/latest"
    ONE_DAY_INTERVAL = "/1d-inteval"
    ASSETS = "/assets"
    REPAY_LOAN = "/assets/repay/{asset}"


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


@router.get(RoutePaths.ASSETS)
async def get_binance_user_assets(
    token: str = Depends(verify_access_token),
):
    with HttpResponse():
        acc_state = get_account_assets_state()
        return {"data": acc_state}
