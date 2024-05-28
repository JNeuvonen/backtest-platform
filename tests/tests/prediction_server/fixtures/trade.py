def create_trade_body(
    open_time_ms: int,
    strategy_id: int,
    open_price: float,
    direction: str,
    quantity: float,
    cumulative_quote_quantity: float,
    symbol: str,
):
    return {
        "open_time_ms": open_time_ms,
        "strategy_id": strategy_id,
        "open_price": open_price,
        "direction": direction,
        "quantity": quantity,
        "cumulative_quote_quantity": cumulative_quote_quantity,
        "symbol": symbol,
    }


def create_trade_close(quantity: float, price: float, close_time_ms: float):
    return {"quantity": quantity, "price": price, "close_time_ms": close_time_ms}


def close_trade_body_test_case_1():
    return create_trade_close(
        quantity=10000 / 65000, price=65000, close_time_ms=1712319993307 + 10000
    )


def trade_test_backend_sanity(strategy_id):
    return create_trade_body(
        open_time_ms=1712319993307,
        open_price=65000,
        strategy_id=strategy_id,
        quantity=10000 / 65000,
        cumulative_quote_quantity=10000,
        direction="LONG",
        symbol="BTCUSDT",
    )
