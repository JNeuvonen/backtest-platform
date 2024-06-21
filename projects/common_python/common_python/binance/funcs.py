from typing import List
import requests
from common_python.binance.client import client
from common_python.binance.types import BinanceUserAsset


SYMBOL_BTCUSDT = "BTCUSDT"
PRICE_KEY = "price"


class MarginAccResKeys:
    TOTAL_ASSET_OF_BTC = "totalAssetOfBtc"
    TOTAL_LIABILITY_OF_BTC = "totalLiabilityOfBtc"
    TOTAL_NET_ASSET_OF_BTC = "totalNetAssetOfBtc"
    MARGIN_LEVEL = "marginLevel"


def get_btc_price():
    btc_price = client.get_symbol_ticker(symbol=SYMBOL_BTCUSDT)[PRICE_KEY]
    return float(btc_price)


def get_binance_acc_balance_snapshot():
    margin_account_info = client.get_margin_account()

    totalAssetInBtc = float(margin_account_info[MarginAccResKeys.TOTAL_ASSET_OF_BTC])
    liabilityInBtc = float(margin_account_info[MarginAccResKeys.TOTAL_LIABILITY_OF_BTC])
    netAssetInBtc = float(margin_account_info[MarginAccResKeys.TOTAL_NET_ASSET_OF_BTC])
    margin_level = float(margin_account_info[MarginAccResKeys.MARGIN_LEVEL])

    btc_price = get_btc_price()

    return {
        "nav_usdt": netAssetInBtc * btc_price,
        "liability_usdt": liabilityInBtc * btc_price,
        "total_asset_usdt": totalAssetInBtc * btc_price,
        "margin_level": margin_level,
        "btc_price": btc_price,
    }


def get_account_assets_state():
    ret = []
    margin_account_info = client.get_margin_account()

    for item in margin_account_info["userAssets"]:
        free = float(item["free"])
        locked = float(item["locked"])
        borrowed = float(item["borrowed"])
        interest = float(item["interest"])
        netAsset = float(item["netAsset"])

        if free != 0 or locked != 0 or borrowed != 0 or interest != 0 or netAsset != 0:
            user_asset = BinanceUserAsset(
                asset=item["asset"],
                borrowed=borrowed,
                locked=locked,
                interest=interest,
                netAsset=netAsset,
                free=free,
            )
            ret.append(user_asset)

    return ret


def get_top_coins_by_usdt_volume(limit=30):
    url = "https://api.binance.com/api/v3/ticker/24hr"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        usdt_pairs = [ticker for ticker in data if ticker["symbol"].endswith("USDT")]

        sorted_by_market_cap = sorted(
            usdt_pairs,
            key=lambda ticker: float(ticker["volume"]) * float(ticker["lastPrice"]),
            reverse=True,
        )

        top_coins = [ticker["symbol"] for ticker in sorted_by_market_cap[:limit]]

        top_coins_filtered = [
            item for item in top_coins if item not in ["USDCUSDT", "FDUSDUSDT"]
        ]

        return top_coins_filtered

    except requests.RequestException as e:
        raise Exception("Failed to fetch data from Binance API") from e


SPOT_EXCHANGE_INFO_ENDPOINT = "https://api.binance.com/api/v3/exchangeInfo"


def get_trade_quantity_precision(symbol: str):
    response = requests.get(SPOT_EXCHANGE_INFO_ENDPOINT)
    data = response.json()

    for item in data["symbols"]:
        if item["symbol"] == symbol:
            for filter in item["filters"]:
                if filter["filterType"] == "LOT_SIZE":
                    min_qty = filter["minQty"]
                    if "." in min_qty:
                        return len(min_qty.split(".")[1].rstrip("0"))
                    else:
                        return 0
    raise ValueError(f"Symbol {symbol} does not exist")


def get_trade_quantities_precision(symbols: List[str]):
    response = requests.get(SPOT_EXCHANGE_INFO_ENDPOINT)
    data = response.json()

    precisions = {}

    for item in data["symbols"]:
        if item["symbol"] in symbols:
            for filter in item["filters"]:
                if filter["filterType"] == "LOT_SIZE":
                    min_qty = filter["minQty"]
                    if "." in min_qty:
                        precisions[item["symbol"]] = len(
                            min_qty.split(".")[1].rstrip("0")
                        )
                    else:
                        precisions[item["symbol"]] = 0

    for symbol in symbols:
        if symbol not in precisions:
            raise ValueError(f"Symbol {symbol} does not exist")
    return precisions
