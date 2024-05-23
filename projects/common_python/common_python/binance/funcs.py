from common_python.binance.client import client


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
    }
