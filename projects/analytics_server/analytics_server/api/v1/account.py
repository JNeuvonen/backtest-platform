from common_python.auth.oauth2 import verify_access_token
from common_python.binance.funcs import repay_loan_with_available_balance
from common_python.http_utils import HttpResponse
from fastapi import APIRouter, Depends


router = APIRouter()


class RoutePaths:
    REPAY_LOAN = "/assets/repay/{asset}"


@router.post(RoutePaths.REPAY_LOAN)
async def post_binance_repay_loan(
    asset: str,
    token: str = Depends(verify_access_token),
):
    with HttpResponse():
        res = repay_loan_with_available_balance(asset)
        return {"data": res}
