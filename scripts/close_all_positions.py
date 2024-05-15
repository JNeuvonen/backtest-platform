import os
import requests
import math
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL
from binance.exceptions import BinanceAPIException, BinanceOrderException

# Endpoint to fetch exchange info
SPOT_EXCHANGE_INFO_ENDPOINT = "https://api.binance.com/api/v3/exchangeInfo"


def get_trade_quantity_precision(symbol):
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


def floor_to_precision(quantity, precision):
    factor = 10**precision
    return math.floor(quantity * factor) / factor


def close_all_margin_positions(client):
    try:
        # Get margin account information to find all open positions
        margin_account_info = client.get_margin_account()
        user_assets = margin_account_info["userAssets"]

        for asset in user_assets:
            free = float(asset["free"])
            borrowed = float(asset["borrowed"])
            net_asset = float(asset["netAsset"])
            if net_asset != 0:
                symbol = asset["asset"]

                if symbol == "USDT":
                    continue

                if net_asset > 0:
                    side = SIDE_SELL
                    quantity = free
                else:
                    side = SIDE_BUY
                    quantity = abs(net_asset)

                symbol_pair = symbol + "USDT"
                precision = get_trade_quantity_precision(symbol_pair)
                quantity = floor_to_precision(quantity, precision)

                print(
                    f"Closing position for {symbol_pair}, Quantity: {quantity}, Side: {side}"
                )

                # Close the position with a market order
                try:
                    order = client.create_margin_order(
                        symbol=symbol_pair,
                        side=side,
                        type="MARKET",
                        quantity=quantity,
                    )
                    print(f"Closed position for {symbol_pair}: {order}")
                except BinanceOrderException as e:
                    print(f"Error closing position for {symbol_pair}: {e}")
                except BinanceAPIException as e:
                    print(f"API error closing position for {symbol_pair}: {e}")

    except BinanceAPIException as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def repay_all_margin_loans(client):
    try:
        # Get margin account information to find all borrowed assets
        margin_account_info = client.get_margin_account()
        user_assets = margin_account_info["userAssets"]

        for asset in user_assets:
            borrowed = float(asset["borrowed"])
            free = float(asset["free"])
            if borrowed > 0:
                asset_name = asset["asset"]
                repay_amount = free

                print(f"Repaying loan for {asset_name}, Amount: {repay_amount}")

                # Repay the loan
                try:
                    repay = client.repay_margin_loan(
                        asset=asset_name, amount=repay_amount
                    )
                    print(f"Repaid loan for {asset_name}: {repay}")
                except BinanceAPIException as e:
                    print(f"API error repaying loan for {asset_name}: {e}")
                except Exception as e:
                    print(f"Unexpected error repaying loan for {asset_name}: {e}")

    except BinanceAPIException as e:
        print(f"API error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    # Load API keys from environment variables
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    if not api_key or not api_secret:
        print("Please set the API_KEY and API_SECRET environment variables.")
        return

    client = Client(api_key, api_secret)

    # Close all margin positions
    close_all_margin_positions(client)

    # Repay all margin loans
    repay_all_margin_loans(client)


if __name__ == "__main__":
    main()
