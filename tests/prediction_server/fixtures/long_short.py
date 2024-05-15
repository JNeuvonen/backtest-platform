from typing import Dict, List
import datetime


transformation_id = 0


def create_long_short_group_body(
    name: str,
    candle_interval: str,
    buy_cond: str,
    sell_cond: str,
    exit_cond: str,
    num_req_klines: int,
    max_simultaneous_positions: int,
    kline_size_ms: int,
    klines_until_close: int,
    max_leverage_ratio: float,
    take_profit_threshold_perc: float,
    stop_loss_threshold_perc: float,
    use_time_based_close: float,
    use_profit_based_close: float,
    use_stop_loss_based_close: float,
    use_taker_order: bool,
    asset_universe: List[Dict],
    data_transformations: List[Dict],
):
    return {
        "name": name,
        "candle_interval": candle_interval,
        "buy_cond": buy_cond,
        "sell_cond": sell_cond,
        "exit_cond": exit_cond,
        "num_req_klines": num_req_klines,
        "max_simultaneous_positions": max_simultaneous_positions,
        "kline_size_ms": kline_size_ms,
        "klines_until_close": klines_until_close,
        "max_leverage_ratio": max_leverage_ratio,
        "take_profit_threshold_perc": take_profit_threshold_perc,
        "stop_loss_threshold_perc": stop_loss_threshold_perc,
        "use_time_based_close": use_time_based_close,
        "use_profit_based_close": use_profit_based_close,
        "use_stop_loss_based_close": use_stop_loss_based_close,
        "use_taker_order": use_taker_order,
        "asset_universe": asset_universe,
        "data_transformations": data_transformations,
    }


BUY_COND_BASIC = """
def get_is_valid_buy(bar):
    # Safely access 'ROCP_7' column, default to None if it does not exist
    value = bar.get('RSI_7_MA_14_close_price')
    rocp_value = bar.get("ROCP_1_close_price")

    if value is None:
        return False
    
    # Check the condition on 'value' values
    if value > 90 and rocp_value < -0.02:
        return True
    return False
"""

SELL_COND_BASIC = """
def get_is_valid_sell(bar):
    # Safely access 'ROCP_7' column, default to None if it does not exist
    value = bar.get('RSI_7_MA_14_close_price')
    rocp_value = bar.get("ROCP_1_close_price")
    
    # Check if 'value' column exists and is not None
    if value is None:
        return False
    # Check the condition on 'value' values
    if value < 10 and rocp_value > 0.02:
        return True
    return False
"""

EXIT_COND_BASIC = """
def get_exit_trade_decision(buy_df, sell_df):
    return False
"""


asset_universe = [
    {"symbol": "ALGOUSDT", "tradeQuantityPrecision": 0},
    {"symbol": "GALAUSDT", "tradeQuantityPrecision": 0},
    {"symbol": "FLOWUSDT", "tradeQuantityPrecision": 2},
    {"symbol": "AXSUSDT", "tradeQuantityPrecision": 2},
    {"symbol": "SANDUSDT", "tradeQuantityPrecision": 0},
    {"symbol": "TRXUSDT", "tradeQuantityPrecision": 1},
    {"symbol": "CVCUSDT", "tradeQuantityPrecision": 0},
    {"symbol": "DEGOUSDT", "tradeQuantityPrecision": 2},
    {"symbol": "DOCKUSDT", "tradeQuantityPrecision": 0},
    {"symbol": "DUSKUSDT", "tradeQuantityPrecision": 0},
]

SMA_CODE_TRANSFORM = """
def calculate_ma(df, column='close_price', periods=[50]):
    for period in periods:
        ma_label = f"MA_{period}_{column}"
        df[ma_label] = df[column].rolling(window=period).mean()

periods = [14]
column = "close_price"
calculate_ma(dataset, column=column, periods=periods)
"""

RSI_CODE_TRANSFORM = """
def calculate_rsi(df, column='open_price', periods=[14]):
    for period in periods:
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df_label = f"RSI_{period}_{column}"
        df[df_label] = 100 - (100 / (1 + rs))

periods = [7]
column = "MA_14_close_price"
calculate_rsi(dataset, column=column, periods=periods)
"""

ROCP_CODE_TRANSFORM = """
import pandas as pd

def calculate_rocp(df, column='close_price', periods=[1]):
    # First, ensure the column exists in DataFrame to avoid KeyError
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

    # Fill NaN values and ensure no infinite values exist in the column
    df[column] = df[column].replace([float('inf'), -float('inf')], pd.NA)
    df[column] = df[column].fillna(method='bfill').fillna(method='ffill')
    
    if df[column].isnull().any():
        raise ValueError("NaN values persist in the column after filling; check data integrity.")

    # Calculate ROCP for each specified period
    for period in periods:
        rocp_label = f"ROCP_{period}_{column}"
        # Shift and calculate difference
        shifted_data = df[column].shift(period)
        differences = df[column] - shifted_data
        # Safeguard against division by zero and inf values
        differences.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        shifted_data.replace(0, pd.NA, inplace=True)  # Replace zero to avoid division by zero

        # Calculate ROCP and handle potential NaN from division
        df[rocp_label] = (differences / shifted_data).fillna(0)

# Usage example:
periods = [1]  # You can specify multiple periods
column = "close_price"
calculate_rocp(dataset, column=column, periods=periods)
"""


def gen_data_transformation_dict(transformation_str):
    global transformation_id
    transformation_id += 1
    current_time = datetime.datetime.now()
    return {
        "id": transformation_id,
        "transformation_code": transformation_str,
        "created_at": current_time.isoformat(),
        "updated_at": current_time.isoformat(),
    }


def long_short_body_basic():
    return create_long_short_group_body(
        name="{SYMBOL}_TEST_LONG_SHORT",
        candle_interval="1d",
        buy_cond=BUY_COND_BASIC,
        sell_cond=SELL_COND_BASIC,
        exit_cond=EXIT_COND_BASIC,
        num_req_klines=60,
        max_simultaneous_positions=65,
        kline_size_ms=86400000,
        klines_until_close=2,
        max_leverage_ratio=1.2,
        take_profit_threshold_perc=5,
        stop_loss_threshold_perc=5,
        use_time_based_close=True,
        use_profit_based_close=False,
        use_stop_loss_based_close=False,
        use_taker_order=True,
        asset_universe=asset_universe,
        data_transformations=[
            gen_data_transformation_dict(SMA_CODE_TRANSFORM),
            gen_data_transformation_dict(RSI_CODE_TRANSFORM),
            gen_data_transformation_dict(ROCP_CODE_TRANSFORM),
        ],
    )


def long_short_test_enter_trade():
    return {
        "buy_open_qty_in_base": 0.6,
        "buy_open_price": 400,
        "sell_open_price": 600,
        "sell_open_qty_in_quote": 600 * 0.4,
        "debt_open_qty_in_base": 0.4,
        "buy_open_time_ms": 1715757773413,
        "sell_open_time_ms": 1715757773413,
        "sell_symbol": "BTCUSDT",
        "buy_symbol": "ETHUSDT",
    }
