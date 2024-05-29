from tests.prediction_server.fixtures.long_short import create_binance_trade_response


ORDER_RES_1 = create_binance_trade_response(
    symbol="BTCUSDT",
    orderId=12345,
    clientOrderId="testOrder123",
    transactTime=1622547801000,
    price="50000.00",
    origQty="0.002",
    executedQty="0.002",
    cummulativeQuoteQty="100.00",
    status="FILLED",
    timeInForce="GTC",
    type="LIMIT",
    side="BUY",
    marginBuyBorrowAmount=50.0,
    marginBuyBorrowAsset="USDT",
    isIsolated=True,
)

ORDER_RES_2 = create_binance_trade_response(
    symbol="ETHUSDT",
    orderId=12346,
    clientOrderId="testOrder124",
    transactTime=1622547901000,
    price="2500.00",
    origQty="0.1",
    executedQty="0.1",
    cummulativeQuoteQty="250.00",
    status="PARTIALLY_FILLED",
    timeInForce="IOC",
    type="MARKET",
    side="SELL",
    marginBuyBorrowAmount=25.0,
    marginBuyBorrowAsset="USDT",
    isIsolated=False,
)
