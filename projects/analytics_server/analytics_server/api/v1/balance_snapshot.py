from common_python.auth.oauth2 import verify_access_token
from common_python.http_utils import HttpResponse
from common_python.pred_serv_models.balance_snapshot import BalanceSnapshotQuery
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    ROOT = "/"


@router.get(RoutePaths.ROOT)
async def get_balance_snapshots(token: str = Depends(verify_access_token)):
    with HttpResponse():
        data = BalanceSnapshotQuery.read_all_entries()
        return {"data": data}
