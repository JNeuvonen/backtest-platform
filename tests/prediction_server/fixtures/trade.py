def create_trade_body(
    open_time_ms: int, strategy_id: int, open_price: float, direction: str
):
    return {
        "open_time_ms": open_time_ms,
        "strategy_id": strategy_id,
        "open_price": open_price,
        "direction": direction,
    }


def trade_test_backend_sanity(strategy_id):
    return create_trade_body(
        open_time_ms=1712319993307,
        open_price=65000,
        strategy_id=strategy_id,
        direction="LONG",
    )
