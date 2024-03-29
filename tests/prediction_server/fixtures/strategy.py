from constants import ONE_DAY_IN_MS


class TradingRules:
    class RSI_30_MA_50_CLOSE_PRICE:
        OPEN = """
        def open_long_trade(tick):
            return tick["RSI_30_MA_50_close_price"] > 95
        """

        CLOSE = """
        def close_long_trade(tick):
            return tick["RSI_30_MA_50_close_price"] < 95
        """

        FETCH = """
        #todo
        print("hello world")
        """

        TRANSFORMATIONS = """
        #todo
        print("hello world")
        """


def create_strategy_body(
    symbol: str,
    enter_trade_code: str,
    exit_trade_code: str,
    fetch_datasources_code: str,
    data_transformations_code: str,
    priority: int,
    kline_size_ms: int,
    klines_left_till_autoclose: int,
    allocated_size_perc: float,
    take_profit_threshold_perc: float,
    stop_loss_threshold_perc: float,
    use_testnet: bool,
    use_time_based_close: bool,
    use_profit_based_close: bool,
    use_stop_loss_based_close: bool,
    use_taker_order: bool,
    is_leverage_allowed: bool,
    is_short_selling_strategy: bool,
):
    return {
        "symbol": symbol,
        "enter_trade_code": enter_trade_code,
        "exit_trade_code": exit_trade_code,
        "fetch_datasources_code": fetch_datasources_code,
        "data_transformations_code": data_transformations_code,
        "priority": priority,
        "kline_size_ms": kline_size_ms,
        "klines_left_till_autoclose": klines_left_till_autoclose,
        "allocated_size_perc": allocated_size_perc,
        "take_profit_threshold_perc": take_profit_threshold_perc,
        "stop_loss_threshold_perc": stop_loss_threshold_perc,
        "use_testnet": use_testnet,
        "use_time_based_close": use_time_based_close,
        "use_profit_based_close": use_profit_based_close,
        "use_stop_loss_based_close": use_stop_loss_based_close,
        "use_taker_order": use_taker_order,
        "is_leverage_allowed": is_leverage_allowed,
        "is_short_selling_strategy": is_short_selling_strategy,
    }


def strategy_simple_1():
    return create_strategy_body(
        symbol="BTCUSDT",
        enter_trade_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.OPEN,
        exit_trade_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.CLOSE,
        fetch_datasources_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.FETCH,
        data_transformations_code=TradingRules.RSI_30_MA_50_CLOSE_PRICE.TRANSFORMATIONS,
        priority=1,
        kline_size_ms=ONE_DAY_IN_MS,
        klines_left_till_autoclose=0,
        allocated_size_perc=100,
        take_profit_threshold_perc=0,
        stop_loss_threshold_perc=0,
        use_testnet=True,
        use_time_based_close=False,
        use_profit_based_close=False,
        use_stop_loss_based_close=False,
        use_taker_order=False,
        is_leverage_allowed=False,
        is_short_selling_strategy=False,
    )
