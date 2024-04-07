def calc_short_trade_perc_result(quantity, open_price, close_price) -> float:
    quote_amount_on_open = quantity * open_price
    quote_amount_on_close = quantity * close_price

    if quote_amount_on_open >= quote_amount_on_close:
        return (quote_amount_on_open / quote_amount_on_close - 1) * 100

    else:
        return (quote_amount_on_close / quote_amount_on_open - 1) * 100


def calc_short_trade_net_result(quantity, open_price, close_price) -> float:
    quote_amount_on_open = quantity * open_price
    quote_amount_on_close = quantity * close_price
    return quote_amount_on_open - quote_amount_on_close


def calc_long_trade_net_result(quantity, open_price, close_price) -> float:
    quote_amount_on_open = quantity * open_price
    quote_amount_on_close = quantity * close_price
    return quote_amount_on_close - quote_amount_on_open


def calc_long_trade_perc_result(quantity, open_price, close_price) -> float:
    quote_amount_on_open = quantity * open_price
    quote_amount_on_close = quantity * close_price
    return (quote_amount_on_close / quote_amount_on_open - 1) * 100
